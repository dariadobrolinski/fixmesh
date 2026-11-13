"""Repair crossings between several surfaces by pushing them apart.

push_apart fixes one surface that crosses itself. Head models are several
surfaces (white matter, gray matter, CSF...) that also cross each other, and
they have to stay separate files because each one is a different tissue.

The surfaces are stacked into one mesh only so one crossing check covers
every pair; no edge joins them, so a push never spreads from one to another.
The push is the same as in push_apart, except for its direction. A wall
normally retreats into its own surface, but that is wrong for the outer of
two nested surfaces: if gray matter pokes through CSF, pulling the CSF in
makes it worse. So which surface sits inside which is worked out first, and:

    self crossing, or surfaces side by side: both walls retreat inward
    one surface inside the other: inner goes inward, outer goes outward

No faces are removed or added, so every surface keeps its own triangles.
Crossings that can't be separated within the movement limit are left in place
and reported; cut_repair isn't used because it works on a single surface.
"""
import argparse
import os
import time

import numpy as np
import open3d as o3d
import pymesh
import trimesh

from .cut import _compute_median_edge_length
from .push_apart import (_cap, _flip_squashed, _min_angles, _outward_sign,
                         _report_triangles, _spread, _vertex_adjacency,
                         _vertex_normals)


# Works out which surface sits inside which. A surface counts as inside
# another when most of its points are, so a brainstem sticking out of the
# bottom of the gray matter doesn't stop white matter counting as inside it.
def _nesting(surfaces):
    """Boolean table: entry [i, j] is True if surface i sits inside j."""
    fraction = np.zeros((len(surfaces), len(surfaces)))
    for j, (vertices, faces) in enumerate(surfaces):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                            o3d.core.Tensor(faces.astype(np.uint32)))
        for i, (points, _) in enumerate(surfaces):
            if i != j:
                fraction[i, j] = scene.compute_occupancy(
                    o3d.core.Tensor(points.astype(np.float32))).numpy().mean()
    return (fraction > 0.5) & (fraction > fraction.T)


# Stacks the surfaces into one mesh. Each face remembers which surface it came
# from, and `starts` says where each surface's points begin.
def _stack(surfaces):
    sizes = [len(vertices) for vertices, _ in surfaces]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    vertices = np.vstack([v for v, _ in surfaces])
    faces = np.vstack([f + start for (_, f), start in zip(surfaces, starts)])
    labels = np.repeat(np.arange(len(surfaces)),
                       [len(f) for _, f in surfaces])
    return vertices, faces, labels, starts


# Sorts the points on crossing faces into those that should move inward and
# those that should move outward. A point on crossings that want opposite
# directions ends up in both, and the two pushes cancel out.
def _push_directions(pairs, faces, labels, inside):
    inward = np.zeros(faces.max() + 1, dtype=bool)
    outward = np.zeros(faces.max() + 1, dtype=bool)
    for own_column, other_column in ((0, 1), (1, 0)):
        own, other = pairs[:, own_column], pairs[:, other_column]
        is_outer = inside[labels[other], labels[own]]
        outward[faces[own[is_outer]].ravel()] = True
        inward[faces[own[~is_outer]].ravel()] = True
    return inward, outward


# Counts crossings per pair of surfaces, for the log.
def _count_by_pair(pairs, labels, names):
    if len(pairs) == 0:
        return {}
    found, counts = np.unique(np.sort(labels[pairs], axis=1), axis=0,
                              return_counts=True)
    return {(f"{names[a]} itself" if a == b else f"{names[a]} / {names[b]}"):
            int(count) for (a, b), count in zip(found, counts)}


def _detect(vertices, faces):
    return np.asarray(pymesh.detect_self_intersection(
        pymesh.form_mesh(vertices, faces))).reshape(-1, 2)


# The main function: sets things up, pushes crossing walls apart round by
# round until nothing crosses (or it gets stuck), and returns the surfaces.
def push_apart_multi_repair(meshes, names=None, step=0.2, rings=3,
                            smoothing_iterations=3, max_displacement=3.0,
                            max_iters=300, patience=15, flip_squashed=True,
                            squashed_angle=2.0, units="mm", verbose=False,
                            output_paths=None):
    """Push surfaces apart until they no longer cross each other or themselves.

    Args:
        meshes (list of pymesh.Mesh): Surfaces to repair, in any order.
        names (list of str): Names for the log. Default: surface 0, 1, ...
        step (float): How far crossing vertices move per iteration, in mm
            (or median edge lengths, see `units`).
        rings (int): How many rings of neighbors share the movement, so the
            surface bends smoothly instead of denting.
        smoothing_iterations (int): How evenly the movement is spread over
            those rings each iteration.
        max_displacement (float): Furthest any vertex may move in total, in
            mm (or median edge lengths). Both walls move, so two surfaces can
            separate by up to twice this.
        max_iters (int): Maximum push iterations.
        patience (int): Stop pushing after this many iterations without a
            new lowest crossing count.
        flip_squashed (bool): Last step: flip the edges of squashed
            triangles so TetGen accepts the surfaces. Nothing moves; a flip
            that makes faces cross is undone.
        squashed_angle (float): A triangle with a corner smaller than this
            many degrees counts as squashed.
        units (str): "mm" (the default) measures `step` and
            `max_displacement` in the mesh's own units, which is mm for brain
            surfaces; "edge" measures them in median edge lengths, for meshes
            in other units.
        verbose (bool): Also print a line for every push round.
        output_paths (list of str): If given, one path per surface to save to.

    Returns:
        list of trimesh.Trimesh: The repaired surfaces, in the input order.
    """
    if units not in ("edge", "mm"):
        raise ValueError(f'units must be "edge" or "mm", not {units!r}')
    if len(meshes) < 2:
        raise ValueError("Expected at least two surfaces; use "
                         "push_apart_repair for a single one.")
    names = names or [f"surface {k}" for k in range(len(meshes))]
    cleaned = [pymesh.remove_duplicated_vertices(mesh)[0] for mesh in meshes]
    surfaces = [(np.asarray(mesh.vertices, dtype=float),
                 np.asarray(mesh.faces, dtype=np.int64)) for mesh in cleaned]

    inside = _nesting(surfaces)
    for i, j in zip(*np.nonzero(inside)):
        print(f"{names[i]} is inside {names[j]}", flush=True)

    original, faces, labels, starts = _stack(surfaces)
    edge = _compute_median_edge_length(pymesh.form_mesh(original, faces))
    scale = edge if units == "edge" else 1.0
    adjacency = _vertex_adjacency(faces, len(original))
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)
    sign = _outward_sign(original, faces)
    limit = max_displacement * scale
    print("Starting push...", flush=True)
    start = time.time()

    displacement = np.zeros_like(original)
    best, best_displacement, stale = None, displacement.copy(), 0
    for iteration in range(1, max_iters + 1):
        vertices = original + displacement
        pairs = _detect(vertices, faces)
        if best is None or len(pairs) < best:
            best, best_displacement, stale = (len(pairs), displacement.copy(),
                                              0)
        else:
            stale += 1
        if iteration == 1:
            initial = _count_by_pair(pairs, labels, names)
            print(f"Crossings at the start: {initial or 'none'}", flush=True)
        if verbose:
            moved = np.linalg.norm(displacement, axis=1)
            print(f"Push {iteration}: {len(pairs):,} crossings "
                  f"{_count_by_pair(pairs, labels, names)}, max move "
                  f"{moved.max():.3f}", flush=True)
        if len(pairs) == 0 or stale >= patience:
            break

        inward, outward = _push_directions(pairs, faces, labels, inside)
        normals = _vertex_normals(vertices, faces) * sign[:, None]
        displacement[inward] -= step * scale * normals[inward]
        displacement[outward] += step * scale * normals[outward]
        # Re-spreads the running total, as in push_apart_repair.
        displacement = _spread(displacement, inward | outward, adjacency,
                               degree, rings, smoothing_iterations)
        displacement = _cap(displacement, limit)

    repaired = original + best_displacement
    moved = np.linalg.norm(best_displacement, axis=1)
    print(f"Pushing done in {time.time() - start:.1f}s ({iteration} rounds).",
          flush=True)
    for k, name in enumerate(names):
        mine = moved[starts[k]:starts[k + 1]]
        shifted = mine[mine > 1e-9]
        print(f"  {name}: {len(shifted):,} of {len(mine):,} vertices moved, "
              f"mean {shifted.mean() if len(shifted) else 0:.3f}, "
              f"max {mine.max():.3f}", flush=True)
    if verbose:
        _report_triangles("Push check", original, repaired, faces)

    if flip_squashed:
        faces = faces.copy()
        squashed = int((_min_angles(repaired, faces) < squashed_angle).sum())
        flipped, undone = _flip_squashed(repaired, faces, squashed_angle)
        left = int((_min_angles(repaired, faces) < squashed_angle).sum())
        print(f"Flipped {flipped:,} edges ({undone} put back because faces "
              f"crossed): {squashed:,} -> {left:,} triangles with a corner "
              f"under {squashed_angle:g} degrees.", flush=True)

    remaining = _count_by_pair(_detect(repaired, faces), labels, names)
    print(f"Crossings left: {remaining or 'none'}", flush=True)

    results = []
    for k in range(len(surfaces)):
        results.append(trimesh.Trimesh(repaired[starts[k]:starts[k + 1]],
                                       faces[labels == k] - starts[k],
                                       process=False))
    if output_paths:
        for result, path in zip(results, output_paths):
            result.export(path)
    return results


# The `push_apart_multi` terminal command: reads the surfaces, repairs them
# and saves each one to its own file. Run `push_apart_multi -h` for options.
def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="push_apart_multi",
        description="Fix crossings between surfaces by pushing them apart.")
    parser.add_argument("inputs", nargs="+",
                        help="Two or more surfaces (.stl, .ply, .obj, ...), "
                             "in any order.")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Folder to save into, as push_apart_<input "
                             "name>.")
    parser.add_argument("-s", "--step", type=float, default=0.2,
                        help="How far crossing points move per round, in mm "
                             "(default 0.2).")
    parser.add_argument("-l", "--limit", type=float, default=3.0,
                        help="Furthest any point may move in total, in mm "
                             "(default 3).")
    parser.add_argument("-e", "--edge-units", action="store_true",
                        help="Measure --step and --limit in triangle sides "
                             "instead of mm (for meshes not in mm).")
    parser.add_argument("--no-flip", action="store_true",
                        help="Don't flip the edges of squashed triangles at "
                             "the end.")
    parser.add_argument("--squashed-angle", type=float, default=2.0,
                        metavar="DEG",
                        help="Triangles with a corner under DEG degrees get "
                             "flipped (default 2).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print a line for every round, not just the "
                             "summaries.")
    args = parser.parse_args(argv)

    if len(args.inputs) < 2:
        parser.error("give at least two surfaces; use push_apart for one")
    for path in args.inputs:
        if not os.path.isfile(path):
            parser.error(f"input file not found: {path}")
    base_names = [os.path.basename(path) for path in args.inputs]
    if len(set(base_names)) < len(base_names):
        parser.error("inputs need different file names, or they would "
                     "overwrite each other in the output folder")
    os.makedirs(args.output_dir, exist_ok=True)
    push_apart_multi_repair(
        [pymesh.load_mesh(path) for path in args.inputs],
        names=[os.path.splitext(name)[0] for name in base_names],
        step=args.step, max_displacement=args.limit,
        flip_squashed=not args.no_flip, squashed_angle=args.squashed_angle,
        units="edge" if args.edge_units else "mm", verbose=args.verbose,
        output_paths=[os.path.join(args.output_dir, "push_apart_" + name)
                      for name in base_names])
    print(f"Saved to {args.output_dir}")
    print("Finished!")


if __name__ == '__main__':
    main()
