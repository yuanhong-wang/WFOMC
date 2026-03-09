from enum import Enum

from wfomc._compat import try_import_cython as _try_cy

_std = _try_cy("wfomc.algo.StandardWFOMC", "wfomc.algo.StandardWFOMC")
_fast = _try_cy("wfomc.algo.FastWFOMC", "wfomc.algo.FastWFOMC")
_incr = _try_cy("wfomc.algo.IncrementalWFOMC", "wfomc.algo.IncrementalWFOMC")
_rec = _try_cy("wfomc.algo.RecursiveWFOMC", "wfomc.algo.RecursiveWFOMC")

standard_wfomc = _std.standard_wfomc
fast_wfomc = _fast.fast_wfomc
incremental_wfomc = _incr.incremental_wfomc
recursive_wfomc = _rec.recursive_wfomc

__all__ = [
    "standard_wfomc",
    "fast_wfomc",
    "incremental_wfomc",
    "recursive_wfomc",
]


class Algo(Enum):
    STANDARD = 'standard'
    FAST = 'fast'
    FASTv2 = 'fastv2'
    INCREMENTAL = 'incremental'
    RECURSIVE = 'recursive'

    def __str__(self):
        return self.value
