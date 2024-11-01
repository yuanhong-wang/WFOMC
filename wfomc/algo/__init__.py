from enum import Enum

from .StandardWFOMC import standard_wfomc
from .FastWFOMC import fast_wfomc
from .IncrementalWFOMC import incremental_wfomc
from .RecursiveWFOMC import recursive_wfomc

__all__ = [
    "standard_wfomc",
    "fast_wfomc",
    "incremental_wfomc",
    "recursive_wfomc"
]


class Algo(Enum):
    STANDARD = 'standard'
    FAST = 'fast'
    FASTv2 = 'fastv2'
    INCREMENTAL = 'incremental'
    RECURSIVE = 'recursive'

    def __str__(self):
        return self.value
