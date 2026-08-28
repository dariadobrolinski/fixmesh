import trimesh
import numpy as np
import pymesh
from collections import defaultdict
import pymeshfix
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

def cut_repair(mesh, max_patch_edge_multiplier=1.6, fairing="membrane",
               refine_patches=False, base_widening=0, max_passes=20,
               max_widening=3):
    """
    Removes intersecting faces and fills the open boundaries left behind,
    keeping only the number of major components that existed before cutting.

    A single cut-and-patch pass does not finish the job. The patches are built
    without any knowledge of the rest of the mesh, so a patch can be laid
    straight through nearby cortex - or, when the input has several
    components, straight through a different component, since each one is
    patched in isolation. The cycle is therefore repeated on its own output
    until no self-intersection is left. On a two-hemisphere brain mesh this
    takes about five passes, and the hemispheres stop intersecting each other
    on the second; almost all of the surface loss happens in the first pass.

    Args:
        mesh (pymesh.Mesh): Input mesh, possibly with self-intersections.
        max_patch_edge_multiplier (float): Maximum edge length in newly
            patched regions, expressed as a multiple of the input mesh's
            median edge length. The paper uses 1.6 for its optional
            long edge refinement.
        fairing (str): How to reshape each refined patch, which is otherwise
            left as the flat disc PyMeshFix produced - refinement alone
            changes a patch's triangle count but not its shape.
            "membrane" solves for the minimal surface spanning the hole rim,
            "thin_plate" for the surface that also meets the rim tangentially
            (smoother, but it overshoots on large holes and drives the patch
            through nearby cortex), and "none" leaves the flat disc alone.
        max_passes (int): Upper bound on cut-and-patch passes. With
            base_widening=0 (the default), convergence has taken 14 passes on
            the two-hemisphere test mesh - the tight cut occasionally needs a
            couple of rounds of stall-triggered widening (see max_widening)
            before it settles.
        refine_patches (bool): After everything else converges, re-fair (and
            then re-triangulate to a finer edge length before fairing again)
            the entire accumulated patch - every face introduced by any pass,
            found the same way `fairing` finds it above - as one continuous
            surface. This is meant for a patch spanning a narrow, elongated
            hole, which has almost no interior to average over at the usual
            triangle scale, so membrane fairing there still reads as a sharp
            fold rather than a rounded valley. It is a real cost, not a free
            cleanup pass: membrane fairing is a minimal-surface solve, so it
            shrinks whatever it touches toward less area, and "the entire
            accumulated patch" can be a large share of the cortex, including
            plenty of patches that were never sharp to begin with. Off by
            default; turn it on only if residual sharp folds matter more than
            the shrinkage this causes everywhere it runs. Never touches an
            original, unpatched vertex's position.
        base_widening (int): Rings of extra faces taken around every
            self-intersection, even on a pass that isn't stalled. Removing
            exactly the flagged faces tends to leave a jagged, slit-shaped
            boundary - the self-intersection test is a pairwise, local check,
            not an estimate of the hole's true shape - and patching a slit
            produces a sharp blade no matter how the patch is refined
            afterwards, because the blade is baked into the boundary that
            patching never moves. A rounder starting cut avoids some of that,
            but every extra ring is real tissue removed unconditionally, on
            every cut, so this trades anatomical fidelity for smoothness
            directly: 0 keeps the cut as tight as possible; each ring above 0
            costs surface area and widens gaps between nearby components in
            exchange for fewer sharp folds later.
        max_widening (int): On top of `base_widening`, if a pass still fails
            to make progress against a stubborn intersection, the margin
            grows by a further ring. This caps how many extra rings may be
            added before iteration gives up.

    A tight spot - most often the narrow channel between two components,
    such as the interhemispheric fissure - can need several passes in a row.
    Each of those passes only knows about the patch the previous pass left,
    not the true cortex beneath it, so the patches stack up and the result
    reads as rough or spiky even though it is watertight and intersection
    free. Once the loop converges, every face introduced across all passes
    is therefore re-faired once more as a single patch relative to the
    original, untouched cortex - not the intermediate per-pass rims - which
    removes that compounding. The convergence loop then runs again in case
    that reshaping reopened an intersection.

    Returns:
        trimesh.Trimesh: Repaired, watertight mesh, free of self-intersections
        unless max_passes was reached first.
    """

    if max_patch_edge_multiplier <= 0:
        raise ValueError("max_patch_edge_multiplier must be positive")
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")

    # Removes duplicated vertices once up front.
    # Records the number of components before cutting.
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    count = _count_num_components(mesh)

    # Computes the edge length from the input's median edge length.
    # This ensures that filled patches are refined to the same scale!
    target_edge_length = (
        _compute_median_edge_length(mesh) * max_patch_edge_multiplier
    )

    # Loop to detect intersections and then cut + patch those interesections
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

    # The final state may have come from _refair_accumulated_patch or
    # _refine_accumulated_patch rather than a plain _cut_repair_pass, and
    # only the latter swept degenerate slivers - so do it once more here
    # regardless of which step produced the returned mesh.
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


def _fair_patch_vertices(vertices, faces, patch_faces, mode="membrane",
                         regularisation=1e-8):
    # Moves the position of free points to create smooth, curved surface.
    if mode not in ("membrane", "thin_plate"):
        raise ValueError(f"unknown fairing mode {mode!r}")

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    free = _free_patch_vertices(faces, patch_faces, len(vertices))
    if not np.any(free):
        return vertices

    laplacian = _umbrella_laplacian(faces, len(vertices))
    free_ids = np.flatnonzero(free)
    pinned = np.where(free[:, None], 0.0, vertices)

    if mode == "membrane":
        system = laplacian[free_ids][:, free_ids].tocsc()
        rhs = -(laplacian[free_ids] @ pinned)
    else:
        lhs = laplacian[:, free_ids]
        system = (lhs.T @ lhs).tocsc()
        system += regularisation * sparse.identity(len(free_ids), format="csc")
        rhs = lhs.T @ -(laplacian @ pinned)

    factor = splu(system)
    faired = vertices.copy()
    faired[free_ids] = np.column_stack(
        [factor.solve(rhs[:, axis]) for axis in range(3)]
    )
    shift = np.linalg.norm(faired[free_ids] - vertices[free_ids], axis=1)
    print(
        f"  Faired {len(free_ids)} patch vertices with {mode} "
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
