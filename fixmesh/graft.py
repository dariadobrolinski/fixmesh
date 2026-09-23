"""Cut-and-patch, but reattach the severed gyri instead of binning them.

Same cut as the current method. The difference: after the cut splits the mesh
into pieces, every piece big enough to be real anatomy is assigned to the
nearest hemisphere and handed to PyMeshFix *together with* that hemisphere,
with joincomp=True, so the freed gyrus gets bridged back on instead of thrown
away and capped over.
"""
import sys
import time
import numpy as np
import pymesh
import pymeshfix
import trimesh
from scipy.spatial import cKDTree

INPUT = '/Users/daria/fixmesh/examples/both_factor_0.stl'
OUT = '/Users/daria/fixmesh/examples/gyri-preserved.stl'
MIN_FRAGMENT_FACES = 60
MAX_PASSES = 8


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
    groups = [[body] for body in bodies]
    for fragment in fragments:
        centre = fragment.vertices.mean(axis=0)
        nearest = int(np.argmin([tree.query(centre)[0] for tree in trees]))
        groups[nearest].append(fragment)

    repaired = []
    for index, group in enumerate(groups):
        merged = trimesh.util.concatenate(group)
        fixer = pymeshfix.MeshFix(merged.vertices, merged.faces)
        # joincomp=True is what bridges the freed gyri back onto the body.
        fixer.repair(verbose=False, joincomp=True,
                     remove_smallest_components=False)
        piece = trimesh.Trimesh(vertices=fixer.v, faces=fixer.f, process=False)
        print(f"    hemisphere {index}: {len(group)-1} fragments reattached, "
              f"{len(piece.faces):,} faces out", flush=True)
        repaired.append(piece)

    out = trimesh.util.concatenate(repaired)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    return pymesh.form_mesh(np.asarray(out.vertices), np.asarray(out.faces))


def main():
    mesh = pymesh.load_mesh(INPUT)
    mesh, _ = pymesh.remove_duplicated_vertices(mesh)
    original_faces = mesh.num_faces
    print(f"input: {original_faces:,} faces", flush=True)

    current = mesh
    start = time.time()
    for index in range(1, MAX_PASSES + 1):
        remaining = len(pymesh.detect_self_intersection(current))
        print(f"Pass {index}: {remaining:,} crossings", flush=True)
        if remaining == 0:
            break
        # Later passes only need to tidy small leftovers, so keep everything.
        current = one_pass(current, 2, MIN_FRAGMENT_FACES if index == 1 else 12)
        current, _ = pymesh.remove_duplicated_vertices(current)

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
