"""Repair self-intersections by pushing crossing walls apart, not cutting.

In a tight sulcus the two walls cross because the fold is thinner than the
mesh can represent. Cutting deletes those walls and caps the hole, which is
where the sulcal depth goes. Here no faces are deleted: every vertex on a
crossing face is nudged a small step back into its own gyrus (against its
outward normal), the movement is spread smoothly over a few rings of
neighbors so the surface keeps its shape, and this repeats until nothing
crosses. The fold keeps its depth and only gets slightly wider.

Walls that no longer cross can still rest right against each other, which
looks merged in a viewer. An optional second phase (off by default) keeps
nudging facing walls apart until there is at least `clearance` between them.

Whatever can't be separated within the movement limit is handed to
cut_repair, so it only has to cut a small remainder. Its patches are left
as PyMeshFix makes them unless smooth_patches is on.
"""
import argparse
import contextlib
import io
import os
import time

import numpy as np
import open3d as o3d
import pymesh
import trimesh
from scipy import sparse

from .cut import _compute_median_edge_length, cut_repair


# Builds a table of which points are joined by an edge, so we know each
# point's neighbors.
def _vertex_adjacency(faces, num_vertices):
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    adjacency = sparse.csr_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(num_vertices, num_vertices))
    adjacency.data[:] = 1.0
    return adjacency


# Works out which way is "out" of the brain for each piece of the mesh, so
# pushes always go back into the gyrus even if a piece was saved inside-out.
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


# Finds the direction pointing straight out of the surface at each point.
# We push against this direction, back into the gyrus.
def _vertex_normals(vertices, faces):
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                            vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-12)


# Shares a push with the surrounding rows of points, so the crossing spot
# moves the most and the movement fades out around it. Keeps the wall
# bending smoothly instead of getting dents.
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


# Picks out crossings that appeared somewhere new, i.e. a wall ran into a
# different wall. Old crossings, and ones that just slid over by one
# triangle as the wall moved, don't count. Used for the "new" count in the
# log.
def _new_crossings(pairs, faces, ever_crossing, adjacency):
    """Crossing pairs where at least one of the two faces isn't on or next to
    a vertex that has ever been on a crossing face. A crossing that only
    slid onto the next triangle as a wall moved doesn't count; a wall
    running into a different wall (e.g. both sides of a thin gyrus) does."""
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.asarray(pairs)
    near = ever_crossing | ((adjacency @ ever_crossing.astype(float)) > 0)
    near_face = near[faces].any(axis=1)
    return pairs[~(near_face[pairs[:, 0]] & near_face[pairs[:, 1]])]


# Puts the points around some crossings (plus a small margin) back to where
# they were before the last push. Used by the gap step.
def _undo_around(pairs, faces, adjacency, rings, candidate, safe):
    """Put the vertices of `pairs`, plus a margin of rings, back to `safe`.
    Returns the mask of vertices that were put back."""
    undo = np.zeros(len(candidate), dtype=bool)
    undo[np.unique(faces[np.asarray(pairs).ravel()])] = True
    for _ in range(rings + 1):
        undo |= (adjacency @ undo.astype(float)) > 0
    candidate[undo] = safe[undo]
    return undo


# Counts triangles that are folded over their neighbors (facing the opposite
# way from the triangles around them) and triangles squashed thin (smallest
# angle under 10 degrees). Only used to report on mesh quality.
def _triangle_check(vertices, faces, face_adjacency, min_angle=10.0):
    """Count triangles folded over their neighbors (facing the opposite way
    from the triangles around them) and triangles squashed below
    `min_angle` degrees."""
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    around = np.zeros_like(normals)
    np.add.at(around, face_adjacency[:, 0], normals[face_adjacency[:, 1]])
    np.add.at(around, face_adjacency[:, 1], normals[face_adjacency[:, 0]])
    folded = np.einsum("ij,ij->i", normals, around) < 0
    corners = vertices[faces]
    angles = []
    for k in range(3):
        a = corners[:, (k + 1) % 3] - corners[:, k]
        b = corners[:, (k + 2) % 3] - corners[:, k]
        cos = np.einsum("ij,ij->i", a, b) / np.maximum(
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
        angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    squashed = np.min(angles, axis=0) < min_angle
    return int(folded.sum()), int(squashed.sum())


# Prints the folded and squashed triangle counts before and after, for the
# log.
def _report_triangles(label, before, after, faces):
    face_adjacency = trimesh.Trimesh(before, faces, process=False).face_adjacency
    folded_before, squashed_before = _triangle_check(before, faces, face_adjacency)
    folded_after, squashed_after = _triangle_check(after, faces, face_adjacency)
    print(f"{label}: {folded_after:,} triangles folded over their neighbors "
          f"(was {folded_before:,}), {squashed_after:,} squashed below 10 "
          f"degrees (was {squashed_before:,})", flush=True)


# Pulls back any point that has moved further than the limit (3 mm), so it
# sits exactly at the limit.
def _cap(displacement, limit):
    length = np.linalg.norm(displacement, axis=1)
    too_far = length > limit
    displacement[too_far] *= (limit / length[too_far])[:, None]
    return displacement


# Measures the distance from each point straight across the fold to the
# opposite wall. Only used by the gap step.
def _gap_to_facing_wall(vertices, faces, outward_normals, edge):
    """Distance from each vertex, straight out along its normal, to the
    first wall it hits (inf if none)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                        o3d.core.Tensor(faces.astype(np.uint32)))
    origins = vertices + outward_normals * 1e-4 * edge
    rays = np.hstack([origins, outward_normals]).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()


# The gap step (off by default): nudges walls that are touching until there
# is a small space between them, without ever making them cross.
def _open_gaps(vertices, faces, clearance, step, rings, smoothing_iterations,
               limit, max_iters, patience, verbose=False):
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
        if verbose:
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
            _undo_around(pairs, faces, adjacency, rings, candidate, displacement)
        else:
            print("  Could not find a safe step; keeping the last safe one.",
                  flush=True)
            break
        displacement = candidate
    if verbose:
        _report_triangles("Gap check", original, original + best_displacement,
                          faces)
    return original + best_displacement


# The main function: sets things up, pushes crossing walls apart round by
# round until nothing crosses (or it gets stuck), cuts whatever is left,
# and returns the repaired mesh.
def push_apart_repair(mesh, step=0.2, rings=3, smoothing_iterations=3,
                      max_displacement=3.0, max_iters=300, patience=15,
                      clearance=0, cut_leftovers=True, smooth_patches=False,
                      units="mm", verbose=False, output_path=None):
    """Push crossing walls apart until the mesh no longer self-intersects.

    Args:
        mesh (pymesh.Mesh): Mesh to repair.
        step (float): How far crossing vertices move per iteration, in mm
            (or median edge lengths, see `units`).
        rings (int): How many rings of neighbors share the movement, so the
            surface bends smoothly instead of denting.
        smoothing_iterations (int): How evenly the movement is spread over
            those rings each iteration.
        max_displacement (float): Furthest any vertex may move in total, in
            mm (or median edge lengths). On the brain meshes 3 mm left the
            fewest crossings for cut_repair; lower limits barely changed the
            average move but left more to cut.
        max_iters (int): Maximum push iterations.
        patience (int): Stop pushing after this many iterations without a
            new lowest crossing count.
        clearance (float): Smallest gap left between facing walls once they
            no longer cross, in mm (or median edge lengths), so touching gyri don't
            look merged. 0 (the default) skips this phase.
        cut_leftovers (bool): Run cut_repair on any crossings left after
            pushing.
        smooth_patches (bool): Make cut_repair's patches finer and smooth
            them into the surrounding surface. Off (the default) leaves
            PyMeshFix's flat, coarse patches, which look sharp. Until
            cut_repair's patch detection is fixed, turning this on also
            flattens large areas outside the patches after pushing.
        units (str): "mm" (the default) measures `step`,
            `max_displacement` and `clearance` in the mesh's own units, which
            is mm for brain surfaces; "edge" measures them in median edge
            lengths, for meshes in other units.
        verbose (bool): Also print a line for every push round, gap round
            and cut_repair pass, instead of just the summaries.
        output_path (str): If given, the result is also saved there.

    Returns:
        trimesh.Trimesh: The repaired mesh.
    """
    if units not in ("edge", "mm"):
        raise ValueError(f'units must be "edge" or "mm", not {units!r}')
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    original = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edge = _compute_median_edge_length(mesh)
    scale = edge if units == "edge" else 1.0
    adjacency = _vertex_adjacency(faces, len(original))
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)
    sign = _outward_sign(original, faces)
    limit = max_displacement * scale
    print("Starting push...", flush=True)
    if verbose:
        print(f"Median edge {edge:.3f}; step {step * scale:.3f}, limit "
              f"{limit:.3f}, clearance {clearance * scale:.3f} (mesh units)",
              flush=True)
    start = time.time()

    displacement = np.zeros_like(original)
    best, best_displacement, stale = None, displacement.copy(), 0
    ever_crossing = np.zeros(len(original), dtype=bool)
    new_total = 0
    for iteration in range(1, max_iters + 1):
        vertices = original + displacement
        pairs = pymesh.detect_self_intersection(pymesh.form_mesh(vertices, faces))
        fresh = (_new_crossings(pairs, faces, ever_crossing, adjacency)
                 if iteration > 1 else [])
        # New crossings are simply pushed apart next round like any other.
        new_total += len(fresh)
        remaining = len(pairs)
        crossing = np.zeros(len(original), dtype=bool)
        crossing[np.unique(faces[np.asarray(pairs).ravel()])] = True
        ever_crossing |= crossing
        if best is None or remaining < best:
            best, best_displacement, stale = remaining, displacement.copy(), 0
        else:
            stale += 1
        if iteration == 1:
            initial = remaining
        if verbose:
            moved = np.linalg.norm(displacement, axis=1)
            print(f"Push {iteration}: {remaining:,} crossings ({len(fresh):,} "
                  f"new), {int(np.count_nonzero(moved > 1e-9)):,} vertices "
                  f"moved (max {moved.max():.3f})", flush=True)
        if remaining == 0 or stale >= patience:
            break

        normals = _vertex_normals(vertices, faces) * sign[:, None]
        displacement[crossing] -= step * scale * normals[crossing]
        # This re-spreads the whole running total, not just this round's
        # push. That makes some cleared spots cross again for a round, but it
        # keeps the surface smooth around spots stuck at the limit; spreading
        # only the new push left ~3x more crossings for cut_repair.
        displacement = _spread(displacement, crossing, adjacency, degree,
                               rings, smoothing_iterations)
        displacement = _cap(displacement, limit)

    current = pymesh.form_mesh(original + best_displacement, faces)
    if verbose:
        moved = np.linalg.norm(best_displacement, axis=1)
        print(f"Pushing done in {time.time() - start:.1f}s ({iteration} "
              f"rounds): {initial:,} crossings -> {best:,} left, "
              f"{int(np.count_nonzero(moved > 1e-9)):,} vertices moved, mean "
              f"move {moved[moved > 1e-9].mean() if moved.any() else 0:.3f}, "
              f"max {moved.max():.3f}; {new_total:,} new crossings appeared; "
              f"no faces removed.",
              flush=True)
        _report_triangles("Push check", original, original + best_displacement,
                          faces)

    if best and cut_leftovers:
        print("Cutting remaining intersections...", flush=True)
        cut_options = (dict(fairing="smoothing", refine_patches=True)
                       if smooth_patches else dict(fairing="none"))
        if verbose:
            result = cut_repair(current, **cut_options)
        else:
            with contextlib.redirect_stdout(io.StringIO()) as cut_log:
                result = cut_repair(current, **cut_options)
            if "Reached max_passes" in cut_log.getvalue():
                print("  Warning: cut_repair ran out of passes with "
                      "crossings remaining.", flush=True)
    else:
        result = trimesh.Trimesh(np.asarray(current.vertices),
                                 np.asarray(current.faces), process=False)

    if clearance > 0:
        print("Opening gaps...", flush=True)
        vertices = _open_gaps(
            np.asarray(result.vertices, dtype=float),
            np.asarray(result.faces, dtype=np.int64),
            clearance * scale, step * scale, rings, smoothing_iterations,
            limit, max_iters, patience, verbose)
        result = trimesh.Trimesh(vertices, np.asarray(result.faces),
                                 process=False)
    if output_path:
        result.export(output_path)
    return result


def _output_path(input_path, output):
    """Where to save: next to the input as push_apart_<name> if no output is
    given, inside `output` if it is a folder, otherwise `output` itself."""
    default_name = "push_apart_" + os.path.basename(input_path)
    if output is None:
        return os.path.join(os.path.dirname(input_path), default_name)
    if os.path.isdir(output) or output.endswith(os.sep):
        os.makedirs(output, exist_ok=True)
        return os.path.join(output, default_name)
    folder = os.path.dirname(output)
    if folder:
        os.makedirs(folder, exist_ok=True)
    return output


# The `push_apart` terminal command: reads a mesh, repairs it and saves it.
# Run `push_apart -h` to see all options.
def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="push_apart",
        description="Fix self-intersections by pushing crossing walls apart.")
    parser.add_argument("input", help="Mesh to repair (.stl, .ply, .obj, ...).")
    parser.add_argument(
        "output", nargs="?",
        help="Where to save: a file name, or a folder (saved there as "
             "push_apart_<input name>). Default: next to the input, as "
             "push_apart_<input name>.")
    parser.add_argument("-s", "--step", type=float, default=0.2,
                        help="How far crossing points move per round, in mm "
                             "(default 0.2).")
    parser.add_argument("-l", "--limit", type=float, default=3.0,
                        help="Furthest any point may move in total, in mm "
                             "(default 3).")
    parser.add_argument("-g", "--gaps", type=float, nargs="?", const=0.25,
                        default=0, metavar="MM",
                        help="Turn on the gap step: leave at least MM between "
                             "walls that touch (default 0.25 if no number).")
    parser.add_argument("-e", "--edge-units", action="store_true",
                        help="Measure --step, --limit and --gaps in triangle "
                             "sides instead of mm (for meshes not in mm).")
    parser.add_argument("--no-cut", action="store_true",
                        help="Don't cut the crossings left after pushing.")
    parser.add_argument("--smooth", action="store_true",
                        help="Make the cut patches finer and smooth them "
                             "into the surrounding surface.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print a line for every round, not just the "
                             "summaries.")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        parser.error(f"input file not found: {args.input}")
    output_path = _output_path(args.input, args.output)
    result = push_apart_repair(
        pymesh.load_mesh(args.input), step=args.step,
        max_displacement=args.limit, clearance=args.gaps,
        cut_leftovers=not args.no_cut, smooth_patches=args.smooth,
        units="edge" if args.edge_units else "mm",
        verbose=args.verbose,
        output_path=output_path)
    remaining = len(pymesh.detect_self_intersection(
        pymesh.form_mesh(np.asarray(result.vertices), np.asarray(result.faces))))
    print(f"Saved {output_path}")
    print(f"Self-intersections left: {remaining:,}")
    print("Finished!")


if __name__ == '__main__':
    main()
