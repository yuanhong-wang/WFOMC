"""Unavailable bounded-treewidth solver extension point."""

from __future__ import annotations

from wfomc.algo.core import SolveContext
from .input import TreeDecompositionInput
from wfomc.errors import UnsupportedFeatureError
from wfomc.result import WFOMCResult

def solve(
    algo_input: TreeDecompositionInput,
    context: SolveContext | None = None,
) -> WFOMCResult:
    raise UnsupportedFeatureError(
        "bounded-treewidth algorithm is registered but no decomposition solver is installed"
    )


__all__ = ["solve"]
