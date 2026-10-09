# Repair methods

Results are on `examples/both_factor_0.stl`.

| | Kenichi | Cut | Push apart |
|---|---|---|---|
| **File** | `methods/kenichi.py` | `methods/cut.py` | `methods/push_apart.py` |
| **Crossing faces** | Deleted, once | Deleted, repeated until none are left | Not deleted: pushed back into their own gyrus, a little at a time |
| **Cut-off gyri** | Thrown away | Reattached if big enough | None (nothing is cut) |
| **Holes** | Filled by PyMeshFix | Filled by PyMeshFix, then subdivided and smoothed | None (only if a few crossings are left at the end, the cut method handles those) |
| **Touching walls** | Not handled | Not handled | Nudged apart until there's a small gap |
| **Crossings left** | 1,213 | 0 | 0 |

**Why the cut method fills the sulcus:** it reattaches cut-off gyri.
When a gyrus gets cut loose, PyMeshFix joins it back to the brain by
stretching a big skin of flat triangles from the gyrus to the edge of the
hole, and that skin covers the sulcus. Smoothing then rounds the skin off,
which is why it looks puffy. Kenichi's method throws cut-off gyri away, so
nothing gets stretched across. Tested by elimination: with smoothing off
and only one cut (like Kenichi's), the flat skin is still there; the only
thing left that differs from Kenichi's is the reattachment.

**Why push apart doesn't:** it never deletes the walls. It moves them apart
slightly, so the fold keeps its depth and just gets a little wider.

## How push apart works

When two walls of a sulcus cross, the cut method deletes them and patches
the hole, which flattens the sulcus. Push apart doesn't delete anything.

1. Find the triangles that cross each other.
2. Move them a tiny step back into their own gyrus, so both walls back away
   from each other.
3. Let the nearby surface move a little too, so it bends smoothly instead of
   denting.
4. Repeat until nothing crosses.
5. Keep nudging any walls that are touching until there's a small gap, so
   they don't look merged.
6. If a few crossings are still left, cut just those.

The fold keeps its depth and only gets a little wider.
