import os
import trimesh
import numpy as np
import pymesh
from collections import defaultdict
import pymeshfix
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

def cut_repair(mesh, max_patch_edge_multiplier=1.6, fairing="smoothing",
               refine_patches=False, base_widening=0, max_passes=20,
               max_widening=3, output_directory=None, output_filename=None,
               strategy="geometry_preserving"):
    """
    Repair a mesh by removing intersecting faces and filling the holes.

    The repair repeats until no intersections remain or max_passes is
    reached. It keeps the same number of main components as the input.

    Args:
        mesh (pymesh.Mesh): Mesh to repair.
        max_patch_edge_multiplier (float): Repair scale measured against the
            input mesh's median edge length. It controls patch edge length in
            legacy mode and local movement size in geometry-preserving mode.
        fairing (str): ``"smoothing"`` blends local vertex movement into the
            surrounding surface. ``"none"`` moves only intersecting faces.
        refine_patches (bool): Legacy-only option that makes completed patches
            finer and smooths them together.
        base_widening (int): Extra neighboring rings affected by repair.
            Geometry-preserving mode blends movement across these rings;
            legacy mode removes them.
        max_passes (int): Maximum number of repair passes.
        max_widening (int): Maximum number of stronger local retries when
            repair stalls.
        output_directory (str): Folder to save the result into. Set this
            (with output_filename) to have the repaired mesh written to
            disk before it's returned. Leave both as None to skip saving.
        output_filename (str): File name for the saved result, e.g.
            "repaired.stl". Set this together with output_directory.
        strategy (str): ``"geometry_preserving"`` keeps every original
            component and uses the original curvature to shape new patches.
            ``"legacy"`` uses the previous largest-component cut-and-cap
            behavior.

    Returns:
        trimesh.Trimesh: The repaired mesh.
    """

    if max_patch_edge_multiplier <= 0:
        raise ValueError("max_patch_edge_multiplier must be positive")
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")
    if strategy not in ("geometry_preserving", "legacy"):
        raise ValueError(f"unknown repair strategy {strategy!r}")
    if fairing not in ("smoothing", "none"):
        raise ValueError(f"unknown fairing mode {fairing!r}")
    if bool(output_directory) != bool(output_filename):
        raise ValueError(
            "output_directory and output_filename must be provided together"
        )

    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    target_edge_length = (
        _compute_median_edge_length(mesh) * max_patch_edge_multiplier
    )

    if strategy == "legacy":
        final_mesh = _run_legacy_cut_repair(
            mesh,
            target_edge_length,
            fairing,
            refine_patches,
            base_widening,
            max_passes,
            max_widening,
        )
    else:
        final_mesh = _run_geometry_preserving_repair(
            mesh,
            target_edge_length,
            fairing,
            refine_patches,
            base_widening,
            max_passes,
            max_widening,
        )

    if output_directory and output_filename:
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, output_filename)
        final_mesh.export(output_path)
        print(f"Saved repaired mesh to {output_path}")

    return final_mesh


def _run_legacy_cut_repair(mesh, target_edge_length, fairing,
                           refine_patches, base_widening, max_passes,
                           max_widening):
    """Previous cut-and-cap implementation, kept for comparisons."""
    count = _count_num_components(mesh)
    current = mesh
    previous_intersections = None
    widening = 0
    refaired = False
    patch_refined = False
    for pass_index in range(1, max_passes + 1):
        intersecting_faces = pymesh.detect_self_intersection(current)
        remaining = len(intersecting_faces)
        print(f"Pass {pass_index}: {remaining} self-intersecting face pairs.")
        if remaining == 0:
            if refine_patches and fairing != "none" and not refaired:
                print("  Converged; re-fairing the accumulated patch as one surface.")
                current = _refair_accumulated_patch(current, mesh, fairing)
                refaired = True
                previous_intersections = None
                widening = 0
                continue
            if refine_patches and not patch_refined:
                print("  Converged; refining the accumulated patch to finer resolution.")
                current = _refine_accumulated_patch(
                    current, mesh, target_edge_length, fairing
                )
                patch_refined = True
                previous_intersections = None
                widening = 0
                continue
            break
        if previous_intersections is not None and remaining >= previous_intersections:
            widening += 1
            if widening > max_widening:
                print("Widening exhausted; stopping.")
                break
            print(f"  No progress; widening the cut by {widening} extra ring(s).")
        else:
            widening = 0
        previous_intersections = remaining

        result = _cut_repair_pass(
            current,
            _widen_selection(
                current, intersecting_faces.flatten(), base_widening + widening
            ),
            count,
            target_edge_length,
            fairing,
        )
        current = pymesh.form_mesh(
            np.asarray(result.vertices), np.asarray(result.faces)
        )
        current, _ = pymesh.remove_duplicated_vertices(current)
    else:
        print(f"Reached max_passes={max_passes} with intersections remaining.")

    final_mesh = _pymesh_to_trimesh(current)
    final_mesh.update_faces(final_mesh.nondegenerate_faces())
    final_mesh.remove_unreferenced_vertices()
    return final_mesh


def _run_geometry_preserving_repair(mesh, target_edge_length, fairing,
                                    refine_patches, base_widening,
                                    max_passes, max_widening):
    """Split intersection curves, then separate components without cutting."""
    expected_components = _count_num_components(mesh)
    current = _resolve_and_separate_components(mesh)
    resolved_vertices = np.asarray(current.vertices, dtype=float).copy()
    stalled_passes = 0
    if refine_patches:
        print(
            "  refine_patches is unnecessary for geometry_preserving repair; "
            "no holes are patched."
        )

    for pass_index in range(1, max_passes + 1):
        intersections = pymesh.detect_self_intersection(current)
        remaining = len(intersections)
        print(f"Pass {pass_index}: {remaining} self-intersecting face pairs.")
        if remaining == 0:
            break

        best = None
        best_remaining = remaining
        for step_scale in (1.0, 0.25, 0.0625):
            candidate = _separate_intersecting_components(
                current,
                intersections,
                target_edge_length,
                smooth=(fairing != "none"),
                smoothing_rings=5 + base_widening + stalled_passes,
                strength=(1.0 + 0.5 * stalled_passes) * step_scale,
                reference_vertices=resolved_vertices,
                max_total_displacement=4.0 * target_edge_length,
                verbose=False,
            )
            if _count_internal_intersections(candidate):
                continue
            candidate_remaining = len(
                pymesh.detect_self_intersection(candidate)
            )
            if candidate_remaining < best_remaining:
                best = candidate
                best_remaining = candidate_remaining

        used_rigid_fallback = False
        if best is None:
            for rigid_scale in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
                candidate = _rigid_separation_step(
                    current,
                    intersections,
                    target_edge_length,
                    strength=rigid_scale,
                )
                candidate_remaining = len(
                    pymesh.detect_self_intersection(candidate)
                )
                if candidate_remaining < best_remaining:
                    best = candidate
                    best_remaining = candidate_remaining
                    used_rigid_fallback = True
                if candidate_remaining == 0:
                    break

        if best is None:
            stalled_passes += 1
            if stalled_passes > max_widening:
                raise RuntimeError(
                    "Could not reduce intersections without changing local "
                    "surface geometry"
                )
            print(
                "  No safe step improved the result; increasing separation "
                f"strength to level {stalled_passes}."
            )
            continue

        moved = np.linalg.norm(
            np.asarray(best.vertices) - np.asarray(current.vertices), axis=1
        )
        total_moved = np.linalg.norm(
            np.asarray(best.vertices) - resolved_vertices, axis=1
        )
        method = "rigid component" if used_rigid_fallback else "local"
        print(
            f"  Accepted {method} step: {remaining} -> {best_remaining} "
            f"pairs; moved {int(np.count_nonzero(moved > 1e-12))} vertices "
            f"(step max {moved.max():.3f}, total max "
            f"{total_moved.max():.3f}); no faces were removed."
        )
        current = best
        stalled_passes = 0
    else:
        remaining = len(pymesh.detect_self_intersection(current))
        if remaining:
            raise RuntimeError(
                f"Reached max_passes={max_passes} with {remaining} "
                "self-intersecting face pairs"
            )

    final_remaining = len(pymesh.detect_self_intersection(current))
    if final_remaining:
        raise RuntimeError(
            f"Repair finished with {final_remaining} self-intersecting face pairs"
        )

    final_mesh = _pymesh_to_trimesh(current)
    final_mesh.update_faces(final_mesh.nondegenerate_faces())
    final_mesh.remove_unreferenced_vertices()
    actual_components = current.num_surface_components
    if actual_components != expected_components:
        raise RuntimeError(
            "Repair changed the number of components from "
            f"{expected_components} to {actual_components}"
        )
    if not final_mesh.is_watertight:
        raise RuntimeError("Geometry-preserving repair produced an open mesh")
    return final_mesh


def _mesh_face_components(mesh):
    """Return connected face ids without changing their original order."""
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )
    return trimesh.graph.connected_components(
        tm.face_adjacency,
        nodes=np.arange(mesh.num_faces),
        min_len=1,
    )


def _submesh_from_faces(mesh, face_ids):
    faces = np.asarray(mesh.faces)[np.asarray(face_ids, dtype=np.int64)]
    used, remapped = np.unique(faces, return_inverse=True)
    return pymesh.form_mesh(
        np.asarray(mesh.vertices)[used], remapped.reshape(-1, 3)
    )


def _resolve_and_separate_components(mesh):
    """Resolve each source component independently, preserving its topology."""
    original_components = _mesh_face_components(mesh)
    components = []
    total_internal = 0
    for component_index, face_ids in enumerate(original_components):
        component = _submesh_from_faces(mesh, face_ids)
        internal = len(pymesh.detect_self_intersection(component))
        total_internal += internal
        if internal:
            print(
                f"  Component {component_index}: splitting {internal} "
                "internal intersection pairs without deleting faces."
            )
            component = pymesh.resolve_self_intersection(
                component, engine="igl"
            )
            component, _ = pymesh.remove_duplicated_vertices(component)
            component, _ = pymesh.remove_duplicated_faces(component)
            component, _ = pymesh.remove_isolated_vertices(component)
        if not component.is_closed():
            raise RuntimeError(
                f"Resolved component {component_index} is not closed"
            )
        remaining = len(pymesh.detect_self_intersection(component))
        if remaining:
            raise RuntimeError(
                f"Resolved component {component_index} still has "
                f"{remaining} internal intersections"
            )
        components.append(component)

    result = pymesh.merge_meshes(components)
    component_labels = np.concatenate([
        np.full(component.num_faces, component_index, dtype=float)
        for component_index, component in enumerate(components)
    ])
    result.add_attribute("source_component")
    result.set_attribute("source_component", component_labels)
    print(
        f"  Preserved all {mesh.num_faces} source faces as "
        f"{result.num_faces} triangles after resolving {total_internal} "
        "within-component pairs."
    )
    return result


def _face_component_labels(mesh):
    if "source_component" in mesh.attribute_names:
        labels = mesh.get_attribute("source_component").astype(np.int64)
        components = [
            np.flatnonzero(labels == component_index)
            for component_index in np.unique(labels)
        ]
        return components, labels

    components = _mesh_face_components(mesh)
    labels = np.empty(mesh.num_faces, dtype=np.int64)
    for component_index, face_ids in enumerate(components):
        labels[np.asarray(face_ids, dtype=np.int64)] = component_index
    return components, labels


def _copy_component_labels(source, target):
    if "source_component" in source.attribute_names:
        target.add_attribute("source_component")
        target.set_attribute(
            "source_component",
            source.get_attribute("source_component"),
        )
    return target


def _count_internal_intersections(mesh):
    total = 0
    components, _ = _face_component_labels(mesh)
    for face_ids in components:
        component = _submesh_from_faces(mesh, face_ids)
        total += len(pymesh.detect_self_intersection(component))
    return total


def _rigid_separation_step(mesh, intersections, target_edge_length,
                           strength=1.0):
    """Translate whole components when local movement would fold a surface."""
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    components, face_labels = _face_component_labels(mesh)
    centroids = []
    component_vertices = []
    for face_ids in components:
        vertex_ids = np.unique(faces[np.asarray(face_ids, dtype=np.int64)])
        component_vertices.append(vertex_ids)
        centroids.append(vertices[vertex_ids].mean(axis=0))
    centroids = np.asarray(centroids)

    directions = np.zeros_like(centroids)
    weights = np.zeros(len(centroids), dtype=float)
    for first_face, second_face in np.asarray(intersections, dtype=np.int64):
        first = face_labels[first_face]
        second = face_labels[second_face]
        if first == second:
            continue
        axis = centroids[first] - centroids[second]
        length = np.linalg.norm(axis)
        if length <= 1e-12:
            continue
        axis /= length
        directions[first] += axis
        directions[second] -= axis
        weights[first] += 1.0
        weights[second] += 1.0

    active = weights > 0
    if not np.any(active):
        return mesh
    directions[active] /= weights[active, None]
    lengths = np.linalg.norm(directions, axis=1)
    step = 0.25 * target_edge_length * strength
    directions[active] *= (
        step / np.maximum(lengths[active], 1e-12)
    )[:, None]
    for component_index, vertex_ids in enumerate(component_vertices):
        vertices[vertex_ids] += directions[component_index]
    return _copy_component_labels(
        mesh, pymesh.form_mesh(vertices, faces)
    )


def _vertex_average_matrix(faces, num_vertices):
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.vstack((edges, edges[:, ::-1])), axis=0)
    rows, cols = edges[:, 0], edges[:, 1]
    degree = np.bincount(rows, minlength=num_vertices).astype(float)
    return sparse.coo_matrix(
        (1.0 / np.maximum(degree[rows], 1.0), (rows, cols)),
        shape=(num_vertices, num_vertices),
    ).tocsr()


def _separate_intersecting_components(mesh, intersections, target_edge_length,
                                      smooth=True, smoothing_rings=3,
                                      strength=1.0, reference_vertices=None,
                                      max_total_displacement=np.inf,
                                      verbose=True):
    """Move intersecting sheets apart while keeping their connectivity."""
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    components, face_labels = _face_component_labels(mesh)
    centroids = []
    for face_ids in components:
        vertex_ids = np.unique(faces[np.asarray(face_ids, dtype=np.int64)])
        centroids.append(vertices[vertex_ids].mean(axis=0))
    centroids = np.asarray(centroids)
    face_centers = vertices[faces].mean(axis=1)
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    normal_lengths = np.linalg.norm(face_normals, axis=1)
    face_normals /= np.maximum(normal_lengths[:, None], 1e-12)

    displacement = np.zeros_like(vertices)
    contributions = np.zeros(len(vertices), dtype=float)
    cross_component_contributions = np.zeros(len(vertices), dtype=float)
    same_component_contributions = np.zeros(len(vertices), dtype=float)
    same_component_pairs = 0
    for first_face, second_face in np.asarray(intersections, dtype=np.int64):
        first_component = face_labels[first_face]
        second_component = face_labels[second_face]
        if first_component == second_component:
            same_component_pairs += 1
            axis = face_centers[first_face] - face_centers[second_face]
        else:
            axis = centroids[first_component] - centroids[second_component]
        axis_length = np.linalg.norm(axis)
        if axis_length <= 1e-12:
            axis = face_normals[first_face] - face_normals[second_face]
            axis_length = np.linalg.norm(axis)
        if axis_length <= 1e-12:
            continue
        axis /= axis_length

        first_vertices = faces[first_face]
        second_vertices = faces[second_face]
        displacement[first_vertices] += axis
        displacement[second_vertices] -= axis
        contributions[first_vertices] += 1.0
        contributions[second_vertices] += 1.0
        if first_component == second_component:
            same_component_contributions[first_vertices] += 1.0
            same_component_contributions[second_vertices] += 1.0
        else:
            cross_component_contributions[first_vertices] += 1.0
            cross_component_contributions[second_vertices] += 1.0

    seeds = contributions > 0
    if not np.any(seeds):
        raise RuntimeError("Could not find vertices to separate")
    displacement[seeds] /= contributions[seeds, None]
    lengths = np.linalg.norm(displacement, axis=1)
    step = 0.25 * target_edge_length * strength
    displacement[seeds] *= (
        step / np.maximum(lengths[seeds], 1e-12)
    )[:, None]
    only_same_component = (
        (same_component_contributions > 0) &
        (cross_component_contributions == 0)
    )
    displacement[only_same_component] *= 0.25

    if smooth:
        adjacency = _vertex_average_matrix(faces, len(vertices))
        for _ in range(max(0, smoothing_rings)):
            displacement = 0.7 * displacement + 0.3 * (adjacency @ displacement)

    if reference_vertices is not None and np.isfinite(max_total_displacement):
        reference_vertices = np.asarray(reference_vertices, dtype=float)
        proposed_total = vertices + displacement - reference_vertices
        proposed_length = np.linalg.norm(proposed_total, axis=1)
        limited = proposed_length > max_total_displacement
        proposed_total[limited] *= (
            max_total_displacement / proposed_length[limited]
        )[:, None]
        displacement = reference_vertices + proposed_total - vertices

    moved = np.linalg.norm(displacement, axis=1)
    vertices += displacement
    total_moved = (
        np.linalg.norm(vertices - reference_vertices, axis=1)
        if reference_vertices is not None
        else moved
    )
    if verbose:
        print(
            f"  Moved {int(np.count_nonzero(moved > 1e-12))} local vertices "
            f"(median {np.median(moved[moved > 1e-12]):.3f}, "
            f"step max {moved.max():.3f}, total max {total_moved.max():.3f}); "
            f"handled {same_component_pairs} within-component pairs; "
            "no faces were removed."
        )
    return _copy_component_labels(
        mesh, pymesh.form_mesh(vertices, faces)
    )


def _classify_repair_faces(vertices, faces, original_vertices, original_faces, tol=1e-4):
    # Which faces are new since the very first cut, regardless of which
    # pass introduced them or how many times that spot has been repatched.
    tree = cKDTree(original_vertices)
    distance, nearest = tree.query(vertices, k=1)
    matched_vertex = distance < tol
    original_face_set = set(map(tuple, np.sort(original_faces, axis=1)))

    all_matched = matched_vertex[faces].all(axis=1)
    is_original = np.zeros(len(faces), dtype=bool)
    candidates = np.flatnonzero(all_matched)
    nearest_ids = np.sort(nearest[faces[candidates]], axis=1)
    is_original[candidates] = [
        tuple(row) in original_face_set for row in nearest_ids
    ]
    return ~is_original


def _refair_accumulated_patch(current, original_mesh, fairing_mode):
    # Cleans up mesh, removing any spiky/sharp edges.
    vertices = np.asarray(current.vertices, dtype=float)
    faces = np.asarray(current.faces, dtype=np.int64)
    patch_faces = _classify_repair_faces(
        vertices, faces, original_mesh.vertices, original_mesh.faces
    )
    if not np.any(patch_faces):
        return current
    faired = _fair_patch_vertices(vertices, faces, patch_faces, mode=fairing_mode)
    print(f"  Re-faired {int(np.count_nonzero(patch_faces))} accumulated patch faces.")
    result, _ = pymesh.remove_duplicated_vertices(
        pymesh.form_mesh(faired, faces)
    )
    return result


def _refine_accumulated_patch(current, original_mesh, target_edge_length,
                              fairing_mode, shrink=2.0):
    # Chops patches into smaller triangles and refines them.
    vertices = np.asarray(current.vertices, dtype=float)
    faces = np.asarray(current.faces, dtype=np.int64)
    patch_faces = _classify_repair_faces(
        vertices, faces, original_mesh.vertices, original_mesh.faces
    )
    if not np.any(patch_faces):
        return current
    print(f"  {int(np.count_nonzero(patch_faces))} accumulated patch faces "
          f"will be refined and re-faired.")

    fine_target = target_edge_length / shrink
    vertices, faces, patch_faces = _refine_patch_faces(
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
        patch_faces,
        fine_target,
    )
    if fairing_mode != "none":
        vertices = _fair_patch_vertices(
            vertices, faces, patch_faces, mode=fairing_mode
        )
    result, _ = pymesh.remove_duplicated_vertices(
        pymesh.form_mesh(vertices, faces)
    )
    return result


def _widen_selection(mesh, face_ids, rings):
    # Gets a list of self interesecting triangles
    # Gows that selection outward by rings steps.
    # Cuts those flagged triangles.
    # Widening selection allows for a cleaner patch.
    selected = np.zeros(mesh.num_faces, dtype=bool)
    selected[face_ids] = True
    for _ in range(rings):
        touched = np.zeros(mesh.num_vertices, dtype=bool)
        touched[mesh.faces[selected].ravel()] = True
        selected |= touched[mesh.faces].any(axis=1)
    return np.flatnonzero(selected)


def _cut_repair_pass(mesh, intersecting_faces, count, target_edge_length, fairing):
    # 1 cut and patch cycle: drop intersecting faces, refill the holes.

    # Step 1: Remove self-intersecting faces
    unique_faces = np.setdiff1d(np.arange(mesh.num_faces), intersecting_faces)
    face_mask = mesh.faces[unique_faces]
    uniq_verts, remap = np.unique(face_mask, return_inverse=True)
    cut_mesh = pymesh.form_mesh(mesh.vertices[uniq_verts], remap.reshape(-1, 3))

    # Step 2: Split into submeshes (We discard small fragments here).
    submeshes = _pymesh_to_trimesh(cut_mesh).split(only_watertight=False)
    submeshes_sorted = sorted(submeshes, key=lambda m: len(m.vertices), reverse=True)
    submeshes_needed = submeshes_sorted[:count]
    print(f"  Found {len(submeshes)} components after cut. Keeping {count} largest.")

    # Step 3: Fill holes only in needed components
    repaired_components = []
    for sub in submeshes_needed:
        if sub.is_watertight:
            repaired_components.append(sub)
        else:
            mf = pymeshfix.MeshFix(sub.vertices, sub.faces)
            mf.repair(verbose=False, joincomp=True, remove_smallest_components=False)
            v, f = mf.v, mf.f
            filled = trimesh.Trimesh(vertices=v, faces=f, process=False)
            patch_faces = _find_new_faces(filled, sub)
            v, f, patch_faces = _refine_patch_faces(
                filled, patch_faces, target_edge_length
            )
            if fairing != "none":
                v = _fair_patch_vertices(v, f, patch_faces, mode=fairing)
            repaired_components.append(
                trimesh.Trimesh(vertices=v, faces=f, process=False)
            )

    # Step 4: Combine and return
    final_mesh = trimesh.util.concatenate(repaired_components)
    final_mesh.update_faces(final_mesh.nondegenerate_faces())
    final_mesh.remove_unreferenced_vertices()
    return final_mesh


def _canonical_face_rows(mesh, decimals=6):
    # Allows us to identify each triangle
    triangles = np.round(mesh.vertices[mesh.faces], decimals=decimals)
    order = np.lexsort(
        (triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]),
        axis=1,
    )
    triangles = np.take_along_axis(triangles, order[:, :, None], axis=1)
    rows = np.ascontiguousarray(triangles.reshape(len(triangles), -1))
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()


def _find_new_faces(filled, cut_component):
    # Find new triangles and id them/
    original_faces = _canonical_face_rows(cut_component)
    filled_faces = _canonical_face_rows(filled)
    return ~np.isin(filled_faces, original_faces)


def _refine_patch_faces(mesh, patch_faces, target_edge_length, max_iters=20):
    # Subdivides any patch edge longer than the target, conformingly:
    # If an edge is shared by a patch triangle and a non-patch triangle,
    # Both get split, not just one side.
    if not np.any(patch_faces):
        return (
            np.asarray(mesh.vertices, dtype=float),
            np.asarray(mesh.faces, dtype=np.int64),
            np.asarray(patch_faces, dtype=bool),
        )
    if not np.isfinite(target_edge_length) or target_edge_length <= 0:
        raise ValueError("The input mesh must have a positive median edge length")

    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    patch_faces = np.asarray(patch_faces, dtype=bool).copy()
    original_patch_count = int(np.count_nonzero(patch_faces))

    for _ in range(max_iters):
        face_edges = np.stack(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1
        )
        sorted_edges = np.sort(face_edges.reshape(-1, 2), axis=1)
        unique_edges, inverse = np.unique(
            sorted_edges, axis=0, return_inverse=True
        )
        inverse = inverse.reshape(-1, 3)
        edge_lengths = np.linalg.norm(
            vertices[unique_edges[:, 0]] - vertices[unique_edges[:, 1]],
            axis=1,
        )
        patch_incidence = np.bincount(
            inverse.ravel(),
            weights=np.repeat(patch_faces.astype(np.int8), 3),
            minlength=len(unique_edges),
        )
        marked = (patch_incidence > 0) & (edge_lengths > target_edge_length)
        marked_ids = np.flatnonzero(marked)
        if len(marked_ids) == 0:
            break

        marked_edges = unique_edges[marked_ids]
        midpoint_ids = np.arange(
            len(vertices), len(vertices) + len(marked_edges), dtype=np.int64
        )
        edge_midpoint = np.full(len(unique_edges), -1, dtype=np.int64)
        edge_midpoint[marked_ids] = midpoint_ids
        vertices = np.vstack((vertices, vertices[marked_edges].mean(axis=1)))

        refined_faces = []
        refined_patch_mask = []
        for face, edge_ids, is_patch in zip(faces, inverse, patch_faces):
            a, b, c = face
            ab, bc, ca = edge_midpoint[edge_ids]
            split = (ab >= 0, bc >= 0, ca >= 0)

            if split == (False, False, False):
                children = [(a, b, c)]
            elif split == (True, False, False):
                children = [(a, ab, c), (ab, b, c)]
            elif split == (False, True, False):
                children = [(b, bc, a), (bc, c, a)]
            elif split == (False, False, True):
                children = [(c, ca, b), (ca, a, b)]
            elif split == (True, True, False):
                children = [(b, bc, ab), (ab, bc, c), (a, ab, c)]
            elif split == (False, True, True):
                children = [(c, ca, bc), (bc, ca, a), (b, bc, a)]
            elif split == (True, False, True):
                children = [(a, ab, ca), (ca, ab, b), (c, ca, b)]
            else:
                children = [
                    (a, ab, ca),
                    (ab, b, bc),
                    (ca, bc, c),
                    (ab, bc, ca),
                ]

            refined_faces.extend(children)
            refined_patch_mask.extend([is_patch] * len(children))

        faces = np.asarray(refined_faces, dtype=np.int64)
        patch_faces = np.asarray(refined_patch_mask, dtype=bool)
    else:
        raise RuntimeError("Patch refinement did not converge")

    print(
        f"  Refined {original_patch_count} patch faces into "
        f"{np.count_nonzero(patch_faces)} faces."
    )
    return vertices, faces, patch_faces


def _free_patch_vertices(faces, patch_faces, num_vertices):
    # Marks a vertex as free to move only if every triangle touching it is patch material
    inside = np.zeros(num_vertices, dtype=bool)
    inside[faces[patch_faces].ravel()] = True
    outside = np.zeros(num_vertices, dtype=bool)
    outside[faces[~patch_faces].ravel()] = True
    return inside & ~outside


def _umbrella_laplacian(faces, num_vertices):
    # Find the average of free point's neighbours to find new position (to smooth)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.vstack((edges, edges[:, ::-1])), axis=0)
    rows, cols = edges[:, 0], edges[:, 1]
    degree = np.bincount(rows, minlength=num_vertices).astype(float)
    weights = 1.0 / np.maximum(degree[rows], 1.0)
    adjacency = sparse.coo_matrix(
        (weights, (rows, cols)), shape=(num_vertices, num_vertices)
    ).tocsr()
    return adjacency - sparse.identity(num_vertices, format="csr")


def _fair_patch_vertices(vertices, faces, patch_faces, mode="smoothing"):
    # Moves the position of free points to create smooth, curved surface.
    if mode != "smoothing":
        raise ValueError(f"unknown fairing mode {mode!r}")

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    free = _free_patch_vertices(faces, patch_faces, len(vertices))
    if not np.any(free):
        return vertices

    laplacian = _umbrella_laplacian(faces, len(vertices))
    free_ids = np.flatnonzero(free)
    pinned = np.where(free[:, None], 0.0, vertices)

    system = laplacian[free_ids][:, free_ids].tocsc()
    rhs = -(laplacian[free_ids] @ pinned)

    factor = splu(system)
    faired = vertices.copy()
    faired[free_ids] = np.column_stack(
        [factor.solve(rhs[:, axis]) for axis in range(3)]
    )
    shift = np.linalg.norm(faired[free_ids] - vertices[free_ids], axis=1)
    print(
        f"  Faired {len(free_ids)} patch vertices "
        f"(median shift {np.median(shift):.3f}, max {shift.max():.3f})."
    )
    return faired


def _pymesh_to_trimesh(mesh):
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

def _trimesh_to_pymesh(mesh):
    return pymesh.form_mesh(vertices=mesh.vertices, faces=mesh.faces)

def _compute_median_edge_length(mesh):
    face_indices = mesh.faces 
    verts = mesh.vertices

    edge_lengths = []
    unique_edges = set()


    for f in face_indices:
        # Each face has 3 edges: (f[0], f[1]), (f[1], f[2]), (f[2], f[0])
        edges = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
        
        for edge in edges:
            edge = tuple(sorted(edge))
            if edge not in unique_edges:
                unique_edges.add(edge)
                
                # Calculate the edge length
                e_length = np.linalg.norm(verts[edge[0]] - verts[edge[1]])
                edge_lengths.append(e_length)

    return np.median(edge_lengths) if edge_lengths else 0

def _collapse_long_edges(mesh):
    tol = _compute_median_edge_length(mesh) * 1.6
    new_mesh, _ = pymesh.split_long_edges(mesh, tol)
    return new_mesh

def _count_num_components(mesh):
    mesh = _pymesh_to_trimesh(mesh)
    components = mesh.split(only_watertight=True)
    return len(components)


def _cut_repair_legacy(mesh):
    """
    Fix self-intersections by cutting the given mesh.

    Args:
        mesh (pymesh.Mesh): The input mesh with self-intersections.

    Returns:
        trimesh.Trimesh: The repaired mesh with no self-intersections.
    """
    # 1) Remove all self‑intersecting faces
    count = _count_num_components(mesh)
    intersecting = pymesh.detect_self_intersection(mesh).flatten()
    intersected = set(intersecting.tolist())

    all_faces   = np.arange(mesh.num_faces)
    keep_faces  = np.setdiff1d(all_faces, list(intersected))
    kept_verts  = mesh.faces[keep_faces]
    uniq_v, remap = np.unique(kept_verts, return_inverse=True)
    new_mesh   = pymesh.form_mesh(mesh.vertices[uniq_v],
                                  remap.reshape(-1,3))

    # 2) Split into  submeshes
    submeshes = _pymesh_to_trimesh(new_mesh).split(only_watertight=False)
    submeshes_sorted = sorted(submeshes,
                              key=lambda m: len(m.vertices),
                              reverse=True)
    submeshes_needed = submeshes_sorted[:count]

    # 3) Process each component
    repaired_submesh_needed = []
    for sub in submeshes_needed:
        # — a) remove one‑ring neighbor faces
        faces = sub.faces
        edge_to_faces = defaultdict(list)
        for fi, f in enumerate(faces):
            for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                e = tuple(sorted((u, v)))
                edge_to_faces[e].append(fi)

        # find all boundary faces (edges used by exactly one face)
        boundary_faces = {fs[0]
                          for e, fs in edge_to_faces.items()
                          if len(fs) == 1}

        # collect neighbors of those boundary faces
        neighbors = set()
        for bf in boundary_faces:
            f = faces[bf]
            for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                e = tuple(sorted((u, v)))
                neighbors.update(edge_to_faces[e])

        # removal set: all neighbor faces (includes the boundary faces themselves)
        to_remove = neighbors

        # build filtered face list
        keep_idx = [i for i in range(len(faces)) if i not in to_remove]
        filtered_faces = faces[keep_idx]

        # re‑index and form a small pymesh.Mesh for this component
        uv, remap_f = np.unique(filtered_faces, return_inverse=True)
        pm = pymesh.form_mesh(sub.vertices[uv],
                              remap_f.reshape(-1, 3))

        # — b) convex hull + c) collapse long edges
        pm = pymesh.convex_hull(pm)
        pm = _collapse_long_edges(pm)

        # back to Trimesh and collect
        repaired_submesh_needed.append(_pymesh_to_trimesh(pm))

    # 4) Reassemble all components
    final = trimesh.util.concatenate(repaired_submesh_needed)
    return final
