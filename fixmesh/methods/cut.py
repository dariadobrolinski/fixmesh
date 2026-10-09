import os
import trimesh
import numpy as np
import pymesh
import pymeshfix
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

def cut_repair(mesh, max_patch_edge_multiplier=1.6, fairing="smoothing",
               refine_patches=False, base_widening=0, max_passes=20,
               max_widening=3, output_directory=None, output_filename=None,
               min_fragment_faces=60):
    """
    Repair a mesh by removing intersecting faces and filling the holes.

    The repair repeats until no intersections remain or max_passes is
    reached. It keeps the same number of main components as the input.

    Args:
        mesh (pymesh.Mesh): Mesh to repair.
        max_patch_edge_multiplier (float): Patch edge length, measured
            against the input mesh's median edge length.
        fairing (str): ``"smoothing"`` smooths each patch into the
            surrounding surface (the flattest surface that closes the hole).
            ``"none"`` leaves PyMeshFix's patch as it is.
        refine_patches (bool): Makes completed patches finer and smooths
            them together.
        base_widening (int): Extra rings of neighboring faces removed
            around every cut.
        max_passes (int): Maximum number of repair passes.
        max_widening (int): Maximum number of stronger local retries when
            repair stalls.
        output_directory (str): Folder to save the result into. Set this
            (with output_filename) to have the repaired mesh written to
            disk before it's returned. Leave both as None to skip saving.
        output_filename (str): File name for the saved result, e.g.
            "repaired.stl". Set this together with output_directory.
        min_fragment_faces (int): Smallest offcut, measured in
            faces, that counts as anatomy rather than debris. Cutting the
            intersecting faces severs gyri from the surface; anything this
            size or larger is reattached to the body it came from instead of
            being discarded. Set to 0 for the old keep-the-largest-components
            behavior.

    Returns:
        trimesh.Trimesh: The repaired mesh.
    """

    if max_patch_edge_multiplier <= 0:
        raise ValueError("max_patch_edge_multiplier must be positive")
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")
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

    final_mesh = _run_cut_repair(
        mesh,
        target_edge_length,
        fairing,
        refine_patches,
        base_widening,
        max_passes,
        max_widening,
        min_fragment_faces,
    )

    if output_directory and output_filename:
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, output_filename)
        final_mesh.export(output_path)
        print(f"Saved repaired mesh to {output_path}")

    return final_mesh


def _run_cut_repair(mesh, target_edge_length, fairing,
                    refine_patches, base_widening, max_passes,
                    max_widening, min_fragment_faces=0):
    """Cut the intersecting faces and patch the holes, until none are left."""
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
            min_fragment_faces,
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


def _regroup_with_fragments(submeshes_sorted, count, min_fragment_faces):
    """Decide which offcuts to keep, and which body each one belongs to.

    Cutting the intersecting faces does not just open holes - it severs the
    mesh, and a gyrus tangled at its base comes away as its own piece. Keeping
    only the largest components throws those gyri out and caps the stump, which
    is where most of the lost anatomy goes. Instead every offcut big enough to
    be anatomy rather than debris is handed to PyMeshFix together with the body
    it came from, so the gyrus gets bridged back on. The bridge counts as new
    patch geometry, so it is subdivided and faired like any other patch.
    """
    bodies, offcuts = submeshes_sorted[:count], submeshes_sorted[count:]
    if min_fragment_faces <= 0:
        print(
            f"  Found {len(submeshes_sorted)} components after cut. "
            f"Keeping {count} largest."
        )
        return bodies

    fragments = [m for m in offcuts if len(m.faces) >= min_fragment_faces]
    dropped = sum(len(m.faces) for m in offcuts if len(m.faces) < min_fragment_faces)
    if not fragments:
        print(
            f"  Found {len(submeshes_sorted)} components after cut. "
            f"Keeping {count} largest; {dropped} offcut faces were debris."
        )
        return bodies

    trees = [cKDTree(body.vertices) for body in bodies]
    groups = [[body] for body in bodies]
    for fragment in fragments:
        centre = fragment.vertices.mean(axis=0)
        nearest = int(np.argmin([tree.query(centre)[0] for tree in trees]))
        groups[nearest].append(fragment)

    print(
        f"  Found {len(submeshes_sorted)} components after cut. Keeping "
        f"{count} largest plus {len(fragments)} severed pieces "
        f"({sum(len(m.faces) for m in fragments)} faces) to reattach; "
        f"{dropped} offcut faces were debris."
    )
    return [
        group[0] if len(group) == 1 else trimesh.util.concatenate(group)
        for group in groups
    ]


def _cut_repair_pass(mesh, intersecting_faces, count, target_edge_length,
                     fairing, min_fragment_faces=0):
    # 1 cut and patch cycle: drop intersecting faces, refill the holes.

    # Step 1: Remove self-intersecting faces
    unique_faces = np.setdiff1d(np.arange(mesh.num_faces), intersecting_faces)
    face_mask = mesh.faces[unique_faces]
    uniq_verts, remap = np.unique(face_mask, return_inverse=True)
    cut_mesh = pymesh.form_mesh(mesh.vertices[uniq_verts], remap.reshape(-1, 3))

    # Step 2: Split into submeshes, then decide what to do with the offcuts.
    submeshes = _pymesh_to_trimesh(cut_mesh).split(only_watertight=False)
    submeshes_sorted = sorted(submeshes, key=lambda m: len(m.vertices), reverse=True)
    submeshes_needed = _regroup_with_fragments(
        submeshes_sorted, count, min_fragment_faces
    )

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
    factor = splu(system)
    rhs = -(laplacian[free_ids] @ pinned)

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

def _count_num_components(mesh):
    mesh = _pymesh_to_trimesh(mesh)
    components = mesh.split(only_watertight=True)
    return len(components)
