"""Unavailable bounded-treewidth solver extension point."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .input import TreeDecompositionInput
from wfomc.errors import UnsupportedFeatureError
from wfomc.result import WFOMCResult

if TYPE_CHECKING:
    from wfomc.engine.runtime import RuntimeContext


def solve(
    algo_input: TreeDecompositionInput,
    runtime: "RuntimeContext | None" = None,
) -> WFOMCResult:
    raise UnsupportedFeatureError(
        "bounded-treewidth algorithm is registered but no decomposition solver is installed"
    )


__all__ = ["solve"]
