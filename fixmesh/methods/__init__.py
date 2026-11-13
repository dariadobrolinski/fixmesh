"""One file per repair method, so they can be compared side by side.

kenichi      -- Kenichi's original cut_repair (cut, keep largest, PyMeshFix fill)
cut          -- cut_repair: cut the crossing faces, reattach cut-off gyri,
                patch the holes with PyMeshFix, repeat until clean
push_apart   -- push crossing walls apart instead of cutting; cut leftovers
push_apart_multi -- push_apart across several surfaces, kept as separate files
detach       -- Kenichi's original detach_repair (shrink inner mesh)
"""
from .cut import cut_repair
from .detach import detach_repair
from .push_apart import push_apart_repair
from .push_apart_multi import push_apart_multi_repair
from .kenichi import cut_repair as kenichi_cut_repair

__all__ = ["cut_repair", "detach_repair", "kenichi_cut_repair",
           "push_apart_repair", "push_apart_multi_repair"]
