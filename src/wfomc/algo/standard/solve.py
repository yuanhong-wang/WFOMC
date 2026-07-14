"""Direct standard WFOMC algorithm over framework cell-graph inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wfomc.cell_graph import CellEvidenceAllocation, CellGraphComponent
from .input import StandardInput
from wfomc.result import WFOMCResult

if TYPE_CHECKING:
    from wfomc.engine.runtime import RuntimeContext
from wfomc.multinomial import MultinomialCoefficients


def solve(
    algo_input: StandardInput,
    runtime: "RuntimeContext | None" = None,
) -> WFOMCResult:
    if not isinstance(algo_input, StandardInput):
        raise TypeError("standard algorithm expects a StandardInput")

    MultinomialCoefficients.setup(algo_input.domain_size)
    arithmetic = algo_input.arithmetic
    result = arithmetic.zero()
    for component in algo_input.components:
        allocation = component.cell_evidence_allocation
        if allocation is None:
            allocation = CellEvidenceAllocation.unconstrained(
                len(component.cells),
                algo_input.domain_size,
            )

        subtotal = arithmetic.zero()
        for config, coefficient in allocation.iter_config_coefficients(arithmetic):
            subtotal = arithmetic.add(
                subtotal,
                arithmetic.multiply(
                    coefficient,
                    _config_weight(component, config, arithmetic),
                ),
            )
        result = arithmetic.add(
            result,
            arithmetic.multiply(component.graph_weight, subtotal),
        )

    return WFOMCResult(result)


def _config_weight(
    component: CellGraphComponent,
    config: tuple[int, ...],
    arithmetic,
) -> object:
    result = arithmetic.one()
    for i, count_i in enumerate(config):
        if count_i == 0:
            continue
        result = arithmetic.multiply(
            result,
            arithmetic.power(component.cell_weights[i], count_i),
        )
        result = arithmetic.multiply(
            result,
            arithmetic.power(
                component.pair_weights[i][i],
                count_i * (count_i - 1) // 2,
            ),
        )
        for j in range(i + 1, len(config)):
            count_j = config[j]
            if count_j == 0:
                continue
            result = arithmetic.multiply(
                result,
                arithmetic.power(
                    component.pair_weights[i][j],
                    count_i * count_j,
                ),
            )
    return result


__all__ = ["solve"]
