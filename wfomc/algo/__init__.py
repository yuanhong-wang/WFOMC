from enum import Enum

from .StandardWFOMC import standard_wfomc
from .FastWFOMC import fast_wfomc, fast_wfomc_with_pc
from .IncrementalWFOMC import incremental_wfomc

__all__ = [
    "standard_wfomc",
    "fast_wfomc",
    "fast_wfomc_with_pc",
    "incremental_wfomc",
]


class Algo(Enum):
    STANDARD = 'standard'
    FAST = 'fast'
    FASTv2 = 'fastv2'
    INCREMENTAL = 'incremental'

    def __str__(self):
        return self.value
