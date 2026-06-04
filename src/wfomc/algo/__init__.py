from enum import Enum

from .StandardWFOMC import standard_wfomc
from .FastWFOMC import fast_wfomc
from .IncrementalWFOMC import incremental_wfomc
from .RecursiveWFOMC import recursive_wfomc
from .PropositionalWFOMC import (
    LinearOrderEncoding,
    propositional_wfomc,
    resolve_linear_order_encoding,
)
from .ganak import GanakError, find_ganak

__all__ = [
    "standard_wfomc",
    "fast_wfomc",
    "incremental_wfomc",
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
    RECURSIVE = 'recursive'
    PROPOSITIONAL = 'propositional'

    def __str__(self):
        return self.value
