from enum import Enum

from .FastWFOMC import fast_wfomc
from .IncrementalWFOMC import incremental_wfomc
from .IncrementalWFOMC3 import incremental_wfomc3
from .PropositionalWFOMC import (
    LinearOrderEncoding,
    propositional_wfomc,
    resolve_linear_order_encoding,
)
from .RecursiveWFOMC import recursive_wfomc
from .StandardWFOMC import standard_wfomc
from .ganak import GanakError, find_ganak

__all__ = [
    "standard_wfomc",
    "fast_wfomc",
    "incremental_wfomc",
    "incremental_wfomc3",
    "recursive_wfomc",
    "propositional_wfomc",
    "LinearOrderEncoding",
    "resolve_linear_order_encoding",
    "GanakError",
    "find_ganak",
]


class Algo(Enum):
    STANDARD = 'standard'
    FAST = 'fast'
    FASTv2 = 'fastv2'
    INCREMENTAL = 'incremental'
    INCREMENTAL3 = 'incremental3'
    RECURSIVE = 'recursive'
    PROPOSITIONAL = 'propositional'

    def __str__(self):
        return self.value
