from enum import Enum
from .StandardWFOMC import standard_wfomc
from .FastWFOMC import fast_wfomc
from .IncrementalWFOMC import incremental_wfomc
from .RecursiveWFOMC import recursive_wfomc
from .IncrementalWFOMC3 import incremental_wfomc3

__all__ = [
    "standard_wfomc",
    "fast_wfomc",
    "incremental_wfomc",
    "recursive_wfomc",
    "incremental_wfomc3",
]


class Algo(Enum):
    STANDARD = 'standard'
    FAST = 'fast'
    FASTv2 = 'fastv2'
    INCREMENTAL = 'incremental'
    INCREMENTAL3 = 'incremental3'
    RECURSIVE = 'recursive'

    def __str__(self):
        return self.value
