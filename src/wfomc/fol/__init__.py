"""Public FOL DSL facade.

The root package exports the same public API as ``wfomc.fol.dsl``. Lower-level
syntax nodes remain available from ``wfomc.fol.syntax`` when code wants the data
definitions without constructor helpers.
"""

from __future__ import annotations

from . import dsl, grounding, syntax
from .analysis import constants, predicates
from .dsl import *  # noqa: F401,F403
from .dsl import __all__ as _DSL_ALL


__all__ = [
    *_DSL_ALL,
    "constants",
    "dsl",
    "grounding",
    "predicates",
    "syntax",
]
