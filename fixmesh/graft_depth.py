"""Cut-and-patch, but reattach severed gyri with a depth-preserving zipper
stitch instead of a flat PyMeshFix cap, and pull patch geometry back toward
the original fold depth instead of smoothing it flat.

Same cut as before: delete every self-intersecting face, split into pieces,
classify big-enough leftovers as real anatomy (gyri) rather than noise, and
send each one home to its nearest hemisphere. The difference is *how* it
gets reattached: instead of handing the hemisphere and its fragments to
PyMeshFix's joincomp -- which bridges the gap with whatever flat
triangulation is cheapest, flattening the sulcus in the process -- each
fragment is zipped directly onto the specific hole it came from, boundary
loop to boundary loop, vertex by vertex, so the seam follows the shape of
the gap instead of capping over it. PyMeshFix's joincomp is kept only as a
fallback for fragments the zipper can't confidently match (multi-hole
pieces, no nearby hole, wildly mismatched loop sizes), and every stitch is
checked against its local neighborhood for new self-intersections before
being accepted.

This variant (forked from graft.py, which stays untouched as a known-good
fallback) also replaces plain Laplacian fairing with a depth-aware version:
most patches still come from PyMeshFix's flat cap, not the zipper, and pure
smoothing has no idea a sulcus used to be there -- it just blends the patch
into the flattest possible surface, erasing the fold. `_depth_pull_patch_vertices`
instead pulls each disconnected patch region partway back toward the
original, pre-cut surface (which still traces the fold's true depth even
where it got removed for self-intersecting), capped to a small distance and
checked locally so it can't recreate the tangle it's targeting.
"""
import sys
import time
from collections import defaultdict

import numpy as np
import pymesh
import pymeshfix
import trimesh
from scipy.spatial import cKDTree

# cut.py already solved the "patches come out as giant, coarse triangles"
# problem for its own PyMeshFix-based repairs (refine_patches): find
# whichever faces are new since the original input mesh, smooth them, then
# subdivide any of their edges longer than the surrounding mesh's typical
# edge and smooth again. Reused here rather than reimplemented, since
# graft.py's PyMeshFix fallback hits the exact same issue. Both steps move
# vertices, which can itself open a new crossing -- cut.py only ever runs
# them from inside its own detect-and-cut loop, so a crossing either one
# introduces gets caught by the very next check. main() below follows the
# same pattern rather than running them as a one-shot pass at the end.
try:
    from fixmesh.cut import (
        _classify_repair_faces,
        _compute_median_edge_length,
        _fair_patch_vertices,
        _free_patch_vertices,
        _refine_accumulated_patch,
    )
except ImportError:
    from cut import (
        _classify_repair_faces,
        _compute_median_edge_length,
        _fair_patch_vertices,
        _free_patch_vertices,
        _refine_accumulated_patch,
    )

INPUT = '/Users/daria/fixmesh/examples/both_factor_0.stl'
OUT = '/Users/daria/fixmesh/examples/graft-depth.stl'
MIN_FRAGMENT_FACES = 60
MAX_PASSES = 30


def _boundary_loops(faces):
    """Trace each open-boundary loop of a triangle soup as an ordered list
    of vertex indices, using the directed half-edge order each boundary
    edge appears in its single owning face. Every loop comes out with the
    same winding convention, which is what lets two loops on either side of
    a gap be stitched together with consistent triangle winding just by
    reversing one of them.
    """
    faces = np.asarray(faces)
    if len(faces) == 0:
        return []
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    sorted_edges = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        sorted_edges, axis=0, return_inverse=True, return_counts=True
    )
    boundary = directed[counts[inverse] == 1]

    # A clean loop needs exactly one outgoing and one incoming boundary
    # edge per vertex. Where a hole's boundary touches another hole or is
    # otherwise non-manifold, a plain dict would silently keep only one
    # of the ambiguous edges and could send a trace into a cycle that
    # never revisits its own start, hanging forever. Instead, drop those
    # vertices entirely -- the fragment they belong to just falls back to
    # PyMeshFix instead of being zippered.
    starts, start_counts = np.unique(boundary[:, 0], return_counts=True)
    ends, end_counts = np.unique(boundary[:, 1], return_counts=True)
    ambiguous = set(starts[start_counts > 1].tolist()) | set(ends[end_counts > 1].tolist())

    next_vertex = {
        u: v for u, v in boundary.tolist() if u not in ambiguous and v not in ambiguous
    }

    loops = []
    seen = set()
    for start in next_vertex:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        current = next_vertex.get(start)
        steps = 0
        # Every vertex here has exactly one outgoing edge, so a trace can
        # only ever terminate (hit None), close (hit start), or -- if
        # some case we haven't thought of slips through -- cycle forever;
        # the step cap guarantees termination either way.
        while current is not None and current != start and steps <= len(next_vertex):
            loop.append(current)
            seen.add(current)
            current = next_vertex.get(current)
            steps += 1
        if current == start:
            loops.append(loop)
    return loops


def _arc_length_fractions(vertices, loop):
    """Normalized cumulative arc length around a closed loop, starting at
    0.0 at loop[0] and reaching 1.0 back at loop[0]. Index k is the
    fraction of the loop's perimeter walked before reaching loop[k].
    """
    pts = vertices[loop]
    segment_lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    return cumulative / total if total > 0 else cumulative


def _bridge_loop(vertices, loop_a, loop_b):
    """Zipper two boundary loops together with a ribbon of triangles.

    Correspondence advances by normalized arc-length fraction on each
    side, not raw edge length: the point 30% of the way around loop_a
    always connects near the point 30% of the way around loop_b,
    regardless of how differently the two loops are shaped or sampled.
    Advancing by raw nearest-edge distance instead can let one side
    "outrun" the other when the loops differ in local shape, producing a
    ribbon that twists back over itself.
    """
    # Empirically (checked against real cut boundaries), a hole's loop and
    # the loop on the fragment that fills it already trace in windings
    # that line up directly -- no reversal needed here.
    pts_a, pts_b = vertices[loop_a], vertices[loop_b]
    dist = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1)
    i0, j0 = np.unravel_index(np.argmin(dist), dist.shape)
    loop_a = loop_a[i0:] + loop_a[:i0]
    loop_b = loop_b[j0:] + loop_b[:j0]
    na, nb = len(loop_a), len(loop_b)

    frac_a = _arc_length_fractions(vertices, loop_a)  # length na + 1
    frac_b = _arc_length_fractions(vertices, loop_b)  # length nb + 1

    triangles = []
    i = j = 0
    for _ in range(na + nb):
        a_curr, a_next = loop_a[i % na], loop_a[(i + 1) % na]
        b_curr, b_next = loop_b[j % nb], loop_b[(j + 1) % nb]
        advance_a = i < na and (j >= nb or frac_a[i + 1] <= frac_b[j + 1])
        if advance_a:
            triangles.append((a_curr, a_next, b_curr))
            i += 1
        else:
            triangles.append((a_curr, b_curr, b_next))
            j += 1
    return triangles


def _stitch_fragment(body_vertices, body_loops, fragment):
    """Try to zipper `fragment` onto whichever hole in `body_loops` it came
    from. `body_loops` is the body's *current* boundary loops, computed once
    per hemisphere and kept up to date by the caller (recomputing it from
    scratch per fragment would mean retracing the whole body's boundary,
    which is on the order of the whole hemisphere, once for every one of
    potentially hundreds of fragments).

    Returns (loop_index, candidate_vertices, fragment_faces, bridge_faces,
    pad) on a confident single-loop match -- `loop_index` is which entry in
    `body_loops` was consumed, so the caller can retire it -- or None if the
    caller should fall back to PyMeshFix's joincomp for this fragment
    instead.
    """
    frag_loops = _boundary_loops(fragment.faces)
    if len(frag_loops) != 1 or not body_loops:
        return None
    frag_loop = frag_loops[0]

    frag_pts = fragment.vertices[frag_loop]
    frag_centre = frag_pts.mean(axis=0)
    frag_radius = np.linalg.norm(frag_pts - frag_centre, axis=1).max()
    frag_edge = np.linalg.norm(frag_pts - np.roll(frag_pts, 1, axis=0), axis=1).mean()

    best = None
    for index, loop in enumerate(body_loops):
        pts = body_vertices[loop]
        centre = pts.mean(axis=0)
        radius = np.linalg.norm(pts - centre, axis=1).max()
        dist = np.linalg.norm(centre - frag_centre)
        if best is None or dist < best[0]:
            best = (dist, index, loop, radius)
    dist, loop_index, body_loop, body_radius = best

    # The two loops should be opposite lips of the same gap: their
    # bounding spheres should roughly touch, and neither should dwarf the
    # other.
    if dist > 1.5 * (body_radius + frag_radius):
        return None
    if not (0.4 <= len(frag_loop) / len(body_loop) <= 2.5):
        return None

    offset = len(body_vertices)
    combined_vertices = np.vstack([body_vertices, fragment.vertices])
    fragment_faces = fragment.faces + offset
    frag_loop_abs = [v + offset for v in frag_loop]
    bridge = np.array(
        _bridge_loop(combined_vertices, list(body_loop), frag_loop_abs),
        dtype=np.int64,
    )
    pad = max(frag_edge, 1e-6) * 4
    return loop_index, combined_vertices, fragment_faces, bridge, pad


def _local_faces(vertices, faces, centre, radius):
    if len(faces) == 0:
        return faces
    face_centres = vertices[faces].mean(axis=1)
    return faces[np.linalg.norm(face_centres - centre, axis=1) <= radius]


def _stitch_is_safe(new_vertices, new_faces, contexts, pad):
    """Check the newly added triangles against their local neighborhood
    across every relevant piece of the current scene -- this body's own
    faces so far, plus every other body and fragment, via `contexts` --
    so a stitch that reopens a crossing anywhere nearby gets caught,
    without paying for a whole-mesh self-intersection check per fragment.
    """
    touched = np.unique(new_faces)
    centre = new_vertices[touched].mean(axis=0)
    spread = np.linalg.norm(new_vertices[touched] - centre, axis=1).max()
    radius = spread + pad

    local_vertices = [new_vertices]
    local_faces = [new_faces]
    offset = len(new_vertices)
    for vertices, faces in contexts:
        nearby = _local_faces(vertices, faces, centre, radius)
        if len(nearby) == 0:
            continue
        local_vertices.append(vertices)
        local_faces.append(nearby + offset)
        offset += len(vertices)

    combined_vertices = np.vstack(local_vertices)
    combined_faces = np.vstack(local_faces)
    used, remapped = np.unique(combined_faces, return_inverse=True)
    local = pymesh.form_mesh(combined_vertices[used], remapped.reshape(-1, 3))
    return len(pymesh.detect_self_intersection(local)) == 0


def _reattach_fragments(body, fragments):
    """Reattach `fragments` onto `body`: zipper-stitch whichever ones have
    a confident, safe single-loop match to one of body's holes, and fall
    back to PyMeshFix's joincomp bridge for the rest.

    The safety check only guards against a bridge that folds back onto
    itself or onto a bridge already accepted earlier in this same loop --
    it does *not* check against the other hemisphere or other not-yet-
    processed fragments. In a densely packed region (many small severed
    pieces sitting close together), that broader check rejects nearly
    everything even when each individual bridge is perfectly well formed,
    since almost any new seam grazes a real, unrelated neighbor. PyMeshFix's
    own joincomp bridge doesn't check against that either -- it relies on
    the outer per-pass loop in `main()` to cut away and retry anything that
    slips through. This gives the zipper the same tolerance, rather than
    holding it to a stricter standard that guarantees it never engages.
    """
    vertices = np.asarray(body.vertices, dtype=float)
    faces = np.asarray(body.faces, dtype=np.int64)
    body_loops = _boundary_loops(faces)
    accumulated_bridges = np.empty((0, 3), dtype=np.int64)

    stitched = no_match = unsafe = 0
    leftover = []
    for fragment in fragments:
        result = _stitch_fragment(vertices, body_loops, fragment)
        if result is None:
            no_match += 1
            leftover.append(fragment)
            continue
        loop_index, candidate_vertices, new_faces, bridge, pad = result
        contexts = [(candidate_vertices, accumulated_bridges)]
        if not _stitch_is_safe(candidate_vertices, bridge, contexts, pad):
            unsafe += 1
            leftover.append(fragment)
            continue
        vertices = candidate_vertices
        faces = np.vstack([faces, new_faces, bridge])
        accumulated_bridges = np.vstack([accumulated_bridges, bridge])
        del body_loops[loop_index]
        stitched += 1

    piece = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if leftover:
        merged = trimesh.util.concatenate([piece] + leftover)
        fixer = pymeshfix.MeshFix(merged.vertices, merged.faces)
        # joincomp=True is what bridges any leftover fragments onto the body.
        fixer.repair(verbose=False, joincomp=True, remove_smallest_components=False)
        piece = trimesh.Trimesh(vertices=fixer.v, faces=fixer.f, process=False)

    print(f"    {stitched} zipper-stitched, {no_match} no confident hole, "
          f"{unsafe} rejected (would self-intersect) -> PyMeshFix join, "
          f"{len(piece.faces):,} faces out", flush=True)
    return piece


def one_pass(current, keep_count, min_faces):
    intersecting = pymesh.detect_self_intersection(current).flatten()
    keep = np.setdiff1d(np.arange(current.num_faces), intersecting)
    used, remapped = np.unique(current.faces[keep], return_inverse=True)
    cut_mesh = trimesh.Trimesh(vertices=np.asarray(current.vertices)[used],
                               faces=remapped.reshape(-1, 3), process=False)

    pieces = sorted(cut_mesh.split(only_watertight=False),
                    key=lambda p: len(p.faces), reverse=True)
    bodies, rest = pieces[:keep_count], pieces[keep_count:]
    fragments = [p for p in rest if len(p.faces) >= min_faces]
    dropped = sum(len(p.faces) for p in rest if len(p.faces) < min_faces)
    print(f"  {len(pieces)} pieces: {keep_count} bodies, "
          f"{len(fragments)} fragments kept "
          f"({sum(len(f.faces) for f in fragments):,} faces), "
          f"{dropped:,} faces dropped as noise", flush=True)

    # Send each fragment home to the nearest hemisphere.
    trees = [cKDTree(body.vertices) for body in bodies]
    assigned = [[] for _ in bodies]
    for fragment in fragments:
        centre = fragment.vertices.mean(axis=0)
        nearest = int(np.argmin([tree.query(centre)[0] for tree in trees]))
        assigned[nearest].append(fragment)

    repaired = []
    for index, (body, frags) in enumerate(zip(bodies, assigned)):
        print(f"    hemisphere {index}: {len(frags)} fragment(s) assigned", flush=True)
        repaired.append(_reattach_fragments(body, frags))

    out = trimesh.util.concatenate(repaired)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    return pymesh.form_mesh(np.asarray(out.vertices), np.asarray(out.faces))


def _local_edge_scale(faces, vertices, vertex_ids):
    """Average length of the edges touching each of `vertex_ids`, used to
    scale how far that vertex is allowed to move."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    sums = np.zeros(len(vertices))
    counts = np.zeros(len(vertices))
    np.add.at(sums, edges[:, 0], lengths)
    np.add.at(counts, edges[:, 0], 1)
    np.add.at(sums, edges[:, 1], lengths)
    np.add.at(counts, edges[:, 1], 1)
    counts = np.maximum(counts, 1)
    return (sums / counts)[vertex_ids]


def _patch_components(vertices, faces, patch_faces):
    """Group patch faces into their disconnected regions (sharing an edge),
    so each hole/seam can be pulled and safety-checked independently
    instead of treating every patch on the mesh as one blob.
    """
    patch_ids = np.flatnonzero(patch_faces)
    if len(patch_ids) == 0:
        return []
    sub = trimesh.Trimesh(vertices=vertices, faces=faces[patch_ids], process=False)
    groups = trimesh.graph.connected_components(
        sub.face_adjacency, nodes=np.arange(len(patch_ids)), min_len=1
    )
    return [patch_ids[g] for g in groups]


def _nearby_faces_for_region(face_centre_tree, faces, vertices, region_face_ids,
                             pad_multiplier=3.0):
    """Faces within a small, capped-pull-sized radius of `region_face_ids`,
    found via a spatial index instead of scanning every face in the mesh --
    with tens of thousands of patch faces spread across hundreds of
    separate regions, a linear scan per region is the difference between
    this finishing in seconds and taking the better part of an hour.
    """
    touched = np.unique(faces[region_face_ids])
    centre = vertices[touched].mean(axis=0)
    spread = np.linalg.norm(vertices[touched] - centre, axis=1).max()
    edge_scale = max(_local_edge_scale(faces, vertices, touched).mean(), 1e-6)
    radius = spread + pad_multiplier * edge_scale
    return np.asarray(face_centre_tree.query_ball_point(centre, radius), dtype=np.int64)


def _local_region_is_safe(vertices, faces, region_face_ids, nearby_face_ids):
    """Check one patch region against a precomputed set of nearby faces for
    self-intersections, scoped to just the area a small, capped pull could
    plausibly reach, rather than paying for a whole-mesh check.
    """
    relevant = np.union1d(region_face_ids, nearby_face_ids)
    used, remapped = np.unique(faces[relevant], return_inverse=True)
    local = pymesh.form_mesh(vertices[used], remapped.reshape(-1, 3))
    return len(pymesh.detect_self_intersection(local)) == 0


def _depth_pull_patch_vertices(vertices, faces, patch_faces, reference_vertices,
                               pull_strength=0.5, max_pull_multiplier=2.0):
    """Like cut.py's _fair_patch_vertices, but each free patch vertex is
    also pulled partway toward the nearest point on the ORIGINAL, pre-cut
    surface instead of settling purely into the flattest blend with its
    neighbors. Pure Laplacian smoothing has no idea a sulcus used to be
    there, so it erases it; the original surface still traces the fold's
    true depth even in the region that got cut away for self-intersecting.

    Each disconnected patch region is pulled and validated independently:
    the pull is capped (max_pull_multiplier * local edge length) and the
    region is checked against its local neighborhood for new self-
    intersections; a mesh this size can have thousands of separate patch
    regions (one per PyMeshFix repair), and each safety check has real
    fixed overhead regardless of region size, so this tries the pull once
    per region -- not a shrinking ladder of retries -- and falls straight
    back to plain smoothing (the original, already-safe behavior) for that
    region if that single attempt isn't safe.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    result = _fair_patch_vertices(vertices, faces, patch_faces, mode="smoothing")

    free = _free_patch_vertices(faces, patch_faces, len(vertices))
    if not np.any(free):
        return result

    tree = cKDTree(reference_vertices)
    components = _patch_components(result, faces, patch_faces)
    # Built once from the pre-pull positions: which faces are "nearby" a
    # given region doesn't meaningfully change from the small nudges this
    # function makes, so there's no need to rebuild this per region or per
    # retry -- only the live vertex positions passed into the actual
    # intersection check need to be current.
    face_centre_tree = cKDTree(result[faces].mean(axis=1))
    print(f"  {len(components):,} disconnected patch region(s) to pull.",
          flush=True)

    pulled, fallback = 0, 0
    for count, region_face_ids in enumerate(components, start=1):
        if count % 200 == 0:
            print(f"    ...{count}/{len(components)} regions processed "
                  f"({pulled} pulled, {fallback} fallback so far)",
                  flush=True)
        region_vertex_ids = np.unique(faces[region_face_ids])
        free_in_region = region_vertex_ids[free[region_vertex_ids]]
        if len(free_in_region) == 0:
            continue

        _, nearest = tree.query(result[free_in_region])
        targets = reference_vertices[nearest]
        pull_vec = targets - result[free_in_region]
        max_pull = max_pull_multiplier * _local_edge_scale(faces, result, free_in_region)
        pull_dist = np.linalg.norm(pull_vec, axis=1)
        clamp = np.minimum(1.0, max_pull / np.maximum(pull_dist, 1e-9))
        clamped_pull = clamp[:, None] * pull_vec

        nearby_face_ids = _nearby_faces_for_region(
            face_centre_tree, faces, result, region_face_ids
        )
        candidate = result.copy()
        candidate[free_in_region] = result[free_in_region] + pull_strength * clamped_pull
        if _local_region_is_safe(candidate, faces, region_face_ids, nearby_face_ids):
            result = candidate
            pulled += 1
        else:
            fallback += 1

    print(f"  Depth pull: {pulled} patch region(s) pulled toward original "
          f"depth, {fallback} left as plain smoothing (would "
          "self-intersect).", flush=True)
    return result


def _depth_aware_refair_accumulated_patch(current, original_mesh,
                                          pull_strength=0.5,
                                          max_pull_multiplier=2.0):
    """Like cut.py's _refair_accumulated_patch, but pulls each patch region
    partway back toward the original, pre-repair surface instead of pure
    Laplacian smoothing, so a fold's depth isn't just erased into the
    flattest possible blend. Falls back to plain smoothing, region by
    region, wherever the pull would reintroduce a self-intersection.
    """
    vertices = np.asarray(current.vertices, dtype=float)
    faces = np.asarray(current.faces, dtype=np.int64)
    patch_faces = _classify_repair_faces(
        vertices, faces, original_mesh.vertices, original_mesh.faces
    )
    if not np.any(patch_faces):
        return current
    print(f"  {int(np.count_nonzero(patch_faces)):,} accumulated patch faces "
          "will be pulled toward original depth where safe.", flush=True)
    faired = _depth_pull_patch_vertices(
        vertices, faces, patch_faces,
        np.asarray(original_mesh.vertices, dtype=float),
        pull_strength=pull_strength,
        max_pull_multiplier=max_pull_multiplier,
    )
    result, _ = pymesh.remove_duplicated_vertices(pymesh.form_mesh(faired, faces))
    return result


def main():
    mesh = pymesh.load_mesh(INPUT)
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    original_faces = mesh.num_faces
    print(f"input: {original_faces:,} faces", flush=True)

    target_edge_length = _compute_median_edge_length(mesh)
    current = mesh
    start = time.time()
    previous_remaining = None
    refaired = False
    refined = False
    cleanup_attempts = 0
    max_cleanup_attempts = 6

    for index in range(1, MAX_PASSES + 1):
        remaining = len(pymesh.detect_self_intersection(current))
        print(f"Pass {index}: {remaining:,} crossings", flush=True)

        if remaining == 0 or remaining == previous_remaining:
            # Either fully converged, or stuck at the same residual with
            # nothing left for the cut step to remove -- both are our cue to
            # clean up patch geometry, fairing before refining, same order
            # cut.py uses. Each step resets and loops back to the crossing
            # check above rather than being trusted blindly, so anything it
            # breaks gets caught and re-cut just like any other pass.
            # cleanup_attempts is a hard backstop: no matter what edge case
            # the flag bookkeeping hits, this guarantees the loop can't
            # cycle between fairing/refining forever.
            if not refaired or not refined:
                cleanup_attempts += 1
            if cleanup_attempts > max_cleanup_attempts:
                print(f"  Stalled with {remaining:,} crossings remaining "
                      "after repeated patch cleanup; stopping.", flush=True)
                break
            if not refaired:
                print("  Fairing accumulated patch geometry (depth-aware).",
                      flush=True)
                current = _depth_aware_refair_accumulated_patch(current, mesh)
                refaired = True
                previous_remaining = None
                continue
            if not refined:
                print("  Refining accumulated patch geometry to a finer "
                      "resolution.", flush=True)
                current = _refine_accumulated_patch(
                    current, mesh, target_edge_length, "smoothing"
                )
                refined = True
                previous_remaining = None
                continue
            if remaining == 0:
                break
            print(f"  Stalled with {remaining:,} crossings remaining after "
                  "patch cleanup; stopping.", flush=True)
            break

        previous_remaining = remaining
        faces_before_cut = current.num_faces
        # Later passes only need to tidy small leftovers, so keep everything.
        current = one_pass(current, 2, MIN_FRAGMENT_FACES if index == 1 else 12)
        current, _ = pymesh.remove_duplicated_vertices(current)
        if current.num_faces != faces_before_cut:
            # This pass actually changed the mesh (reattached a fragment,
            # zippered or PyMeshFix-joined something new), so that geometry
            # is fresh and unrefined -- give the next convergence point
            # another chance to catch it. If nothing changed (0 fragments
            # touched -- the residual crossings can't be cut away further),
            # leave the flags alone so a stall that already went through
            # fairing progresses to refining next time, instead of looping
            # back through fairing forever without ever trying refine.
            refaired = False
            refined = False
    else:
        print(f"Reached max_passes={MAX_PASSES} with intersections remaining.",
              flush=True)

    final = trimesh.Trimesh(vertices=np.asarray(current.vertices),
                            faces=np.asarray(current.faces), process=False)
    final.export(OUT)
    crossings = len(pymesh.detect_self_intersection(current))
    print(f"\nDone in {time.time()-start:.1f}s -> {OUT}", flush=True)
    print(f"faces {len(final.faces):,} (input {original_faces:,}) | "
          f"crossings {crossings} | watertight {final.is_watertight} | "
          f"components {final.body_count}", flush=True)


if __name__ == '__main__':
    sys.exit(main())
