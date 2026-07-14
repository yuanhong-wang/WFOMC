"""Direct recursive WFOMC algorithm over framework cell-graph inputs."""

from __future__ import annotations


from typing import TYPE_CHECKING

from .input import RecursiveInput
from wfomc.result import WFOMCResult

if TYPE_CHECKING:
    from wfomc.engine.runtime import RuntimeContext


def solve(
    algo_input: RecursiveInput,
    runtime: "RuntimeContext | None" = None,
) -> WFOMCResult:
    from .kernel import NautyContext, dfs_wfomc_real

    if not isinstance(algo_input, RecursiveInput):
        raise TypeError("recursive algorithm expects a RecursiveInput")

    arithmetic = algo_input.arithmetic
    result = arithmetic.zero()
    for component in algo_input.components:
        cell_weights = list(component.cell_weights)
        pair_weights = [list(row) for row in component.pair_weights]
        nauty_ctx = NautyContext(
            algo_input.domain_size,
            cell_weights,
            pair_weights,
            arithmetic,
        )
        subtotal = dfs_wfomc_real(
            cell_weights,
            pair_weights,
            algo_input.domain_size,
            nauty_ctx,
        )
        result = arithmetic.add(
            result,
            arithmetic.multiply(component.graph_weight, subtotal),
        )

    return WFOMCResult(result)


__all__ = ["solve"]
