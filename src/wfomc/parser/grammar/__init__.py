"""Grammar constants for the framework parsers."""

from .cardinality import CARDINALITY_GRAMMAR
from .fol import FORMULA_GRAMMAR
from .mln import MLN_GRAMMAR
from .wfomcs import DOMAIN_GRAMMAR, WFOMCS_GRAMMAR

__all__ = [
    "CARDINALITY_GRAMMAR",
    "DOMAIN_GRAMMAR",
    "FORMULA_GRAMMAR",
    "MLN_GRAMMAR",
    "WFOMCS_GRAMMAR",
]
