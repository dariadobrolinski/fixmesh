import trimesh
import numpy as np
import pymesh
from collections import defaultdict
import pymeshfix

def cut_repair(mesh):
    """
    Removes intersecting faces and fills open boundary holes.
    Keeps only the number of major components that existed before cutting.

    Args:
        mesh (pymesh.Mesh): Input mesh, possibly with self-intersections.

    Returns:
        trimesh.Trimesh: Repaired, non-intersecting, watertight mesh.
    """

    # Step 0: Record number of connected components before cut
    mesh, _  = pymesh.remove_duplicated_vertices(mesh)
    count = _count_num_components(mesh)

    # Step 1: Remove self-intersecting faces
    intersecting_faces = pymesh.detect_self_intersection(mesh).flatten()
    unique_faces = np.setdiff1d(np.arange(mesh.num_faces), intersecting_faces)
    face_mask = mesh.faces[unique_faces]
    uniq_verts, remap = np.unique(face_mask, return_inverse=True)
    cut_mesh = pymesh.form_mesh(mesh.vertices[uniq_verts], remap.reshape(-1, 3))

    # Step 2: Split into submeshes (We discard small fragments here).
    submeshes = _pymesh_to_trimesh(cut_mesh).split(only_watertight=False)
    submeshes_sorted = sorted(submeshes, key=lambda m: len(m.vertices), reverse=True)
    submeshes_needed = submeshes_sorted[:count]
    print(f"Found {len(submeshes)} components after cut. Keeping {count} largest.")

    # Step 3: Fill holes only in needed components
    repaired_components = []
    for sub in submeshes_needed:
        if sub.is_watertight:
            repaired_components.append(sub)
        else:
            mf = pymeshfix.MeshFix(sub.vertices, sub.faces)
            mf.repair(verbose=False, joincomp=True, remove_smallest_components=False)
            v, f = mf.v, mf.f
            repaired_components.append(trimesh.Trimesh(vertices=v, faces=f, process=False))

    # Step 4: Combine and return
    final_mesh = trimesh.util.concatenate(repaired_components)
    return final_mesh

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
