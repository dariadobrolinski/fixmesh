"""Repair self-intersections by pushing crossing walls apart, not cutting.

In a tight sulcus the two walls cross because the fold is thinner than the
mesh can represent. Cutting deletes those walls and caps the hole, which is
where the sulcal depth goes. Here no faces are deleted: every vertex on a
crossing face is nudged a small step back into its own gyrus (against its
outward normal), the movement is spread smoothly over a few rings of
neighbors so the surface keeps its shape, and this repeats until nothing
crosses. The fold keeps its depth and only gets slightly wider.

Walls that no longer cross can still rest right against each other, which
looks merged in a viewer. A second phase keeps nudging facing walls apart
until there is at least `clearance` between them.

Whatever can't be separated within the movement limit is handed to
cut_repair (legacy, no smoothing), so it only has to cut a small remainder.
"""
import sys
import time

import numpy as np
import open3d as o3d
import pymesh
import trimesh
from scipy import sparse

from .cut import _compute_median_edge_length, cut_repair


def _vertex_adjacency(faces, num_vertices):
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    adjacency = sparse.csr_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(num_vertices, num_vertices))
    adjacency.data[:] = 1.0
    return adjacency


def _outward_sign(vertices, faces):
    """+1 or -1 per vertex, so that sign * normal points out of the brain
    even if a component's faces are wound inside-out."""
    labels = trimesh.graph.connected_component_labels(
        trimesh.Trimesh(vertices, faces, process=False).face_adjacency,
        node_count=len(faces))
    v0, v1, v2 = (vertices[faces[:, k]] for k in range(3))
    signed = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    volume = np.bincount(labels, weights=signed)
    face_sign = np.where(volume[labels] >= 0, 1.0, -1.0)
    sign = np.ones(len(vertices))
    sign[faces.ravel()] = np.repeat(face_sign, 3)
    return sign


def _vertex_normals(vertices, faces):
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                            vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-12)


def _spread(displacement, moving, adjacency, degree, rings, iterations):
    """Share the movement of `moving` vertices with `rings` rings of
    neighbors, so the surface bends smoothly instead of denting."""
    region = moving.copy()
    for _ in range(rings):
        region |= (adjacency @ region.astype(float)) > 0
    for _ in range(iterations):
        averaged = (adjacency @ displacement) / degree[:, None]
        displacement[region] = 0.5 * (displacement[region] + averaged[region])
    return displacement


def _cap(displacement, limit):
    length = np.linalg.norm(displacement, axis=1)
    too_far = length > limit
    displacement[too_far] *= (limit / length[too_far])[:, None]
    return displacement


def _gap_to_facing_wall(vertices, faces, outward_normals, edge):
    """Distance from each vertex, straight out along its normal, to the
    first wall it hits (inf if none)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                        o3d.core.Tensor(faces.astype(np.uint32)))
    origins = vertices + outward_normals * 1e-4 * edge
    rays = np.hstack([origins, outward_normals]).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()


def _open_gaps(vertices, faces, clearance, step, rings, smoothing_iterations,
               limit, max_iters, patience):
    """Push facing walls apart until they are at least `clearance` apart,
    without ever creating a crossing. Each wall closes half the shortfall."""
    original = vertices
    adjacency = _vertex_adjacency(faces, len(original))
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)
    sign = _outward_sign(original, faces)
    displacement = np.zeros_like(original)
    best, best_displacement, stale = None, displacement.copy(), 0
    for iteration in range(1, max_iters + 1):
        current = original + displacement
        normals = _vertex_normals(current, faces) * sign[:, None]
        gap = _gap_to_facing_wall(current, faces, normals, 1.0)
        tight = gap < clearance
        if best is None or tight.sum() < best:
            best, best_displacement, stale = int(tight.sum()), displacement.copy(), 0
        else:
            stale += 1
        print(f"Gap {iteration}: {int(tight.sum()):,} vertices closer than "
              f"{clearance:.3f} to the facing wall", flush=True)
        if not tight.any() or stale >= patience:
            break

        shortfall = np.minimum((clearance - gap[tight]) / 2, step)
        candidate = displacement.copy()
        candidate[tight] -= shortfall[:, None] * normals[tight]
        candidate = _spread(candidate, tight, adjacency, degree, rings,
                            smoothing_iterations)
        candidate = _cap(candidate, limit)
        # Where the step would make walls cross, undo it just around those
        # spots (plus a margin) and keep it everywhere else.
        for _ in range(5):
            pairs = pymesh.detect_self_intersection(
                pymesh.form_mesh(original + candidate, faces))
            if len(pairs) == 0:
                break
            undo = np.zeros(len(original), dtype=bool)
            undo[np.unique(faces[pairs.ravel()])] = True
            for _ in range(rings + 1):
                undo |= (adjacency @ undo.astype(float)) > 0
            candidate[undo] = displacement[undo]
        else:
            print("  Could not find a safe step; keeping the last safe one.",
                  flush=True)
            break
        displacement = candidate
    return original + best_displacement


def push_apart_repair(mesh, step=0.2, rings=3, smoothing_iterations=3,
                      max_displacement=3.0, max_iters=300, patience=15,
                      clearance=0.25, cut_leftovers=True, output_path=None):
    """Push crossing walls apart until the mesh no longer self-intersects.

    Args:
        mesh (pymesh.Mesh): Mesh to repair.
        step (float): How far crossing vertices move per iteration, in
            median edge lengths.
        rings (int): How many rings of neighbors share the movement, so the
            surface bends smoothly instead of denting.
        smoothing_iterations (int): How evenly the movement is spread over
            those rings each iteration.
        max_displacement (float): Furthest any vertex may move in total, in
            median edge lengths.
        max_iters (int): Maximum push iterations.
        patience (int): Stop pushing after this many iterations without a
            new lowest crossing count.
        clearance (float): Smallest gap left between facing walls once they
            no longer cross, in median edge lengths, so touching gyri don't
            look merged. 0 skips this phase.
        cut_leftovers (bool): Run cut_repair (legacy, no smoothing) on any
            crossings left after pushing.
        output_path (str): If given, the result is also saved there.

    Returns:
        trimesh.Trimesh: The repaired mesh.
    """
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    original = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edge = _compute_median_edge_length(mesh)
    adjacency = _vertex_adjacency(faces, len(original))
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)
    sign = _outward_sign(original, faces)
    limit = max_displacement * edge
    start = time.time()

    displacement = np.zeros_like(original)
    best, best_displacement, stale = None, displacement.copy(), 0
    for iteration in range(1, max_iters + 1):
        vertices = original + displacement
        pairs = pymesh.detect_self_intersection(pymesh.form_mesh(vertices, faces))
        remaining = len(pairs)
        if best is None or remaining < best:
            best, best_displacement, stale = remaining, displacement.copy(), 0
        else:
            stale += 1
        moved = np.linalg.norm(displacement, axis=1)
        print(f"Push {iteration}: {remaining:,} crossings, "
              f"{int(np.count_nonzero(moved > 1e-9)):,} vertices moved "
              f"(max {moved.max():.3f})", flush=True)
        if remaining == 0 or stale >= patience:
            break

        crossing = np.zeros(len(original), dtype=bool)
        crossing[np.unique(faces[pairs.ravel()])] = True
        normals = _vertex_normals(vertices, faces) * sign[:, None]
        displacement[crossing] -= step * edge * normals[crossing]
        displacement = _spread(displacement, crossing, adjacency, degree,
                               rings, smoothing_iterations)
        displacement = _cap(displacement, limit)

    current = pymesh.form_mesh(original + best_displacement, faces)
    moved = np.linalg.norm(best_displacement, axis=1)
    print(f"Pushing done in {time.time() - start:.1f}s: {best:,} crossings left, "
          f"{int(np.count_nonzero(moved > 1e-9)):,} vertices moved, "
          f"mean move {moved[moved > 1e-9].mean() if moved.any() else 0:.3f}, "
          f"max {moved.max():.3f}; no faces removed.", flush=True)

    if best and cut_leftovers:
        print("Cutting the remaining crossings with cut_repair.", flush=True)
        result = cut_repair(current, strategy="legacy", fairing="none")
    else:
        result = trimesh.Trimesh(np.asarray(current.vertices),
                                 np.asarray(current.faces), process=False)

    if clearance > 0:
        vertices = _open_gaps(
            np.asarray(result.vertices, dtype=float),
            np.asarray(result.faces, dtype=np.int64),
            clearance * edge, step * edge, rings, smoothing_iterations,
            limit, max_iters, patience)
        result = trimesh.Trimesh(vertices, np.asarray(result.faces),
                                 process=False)
    if output_path:
        result.export(output_path)
    return result


if __name__ == '__main__':
    # python -m fixmesh.methods.push_apart <input> <output>
    push_apart_repair(pymesh.load_mesh(sys.argv[1]), output_path=sys.argv[2])
