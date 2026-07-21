"""Public solver entry point for Boundary-Profile DP."""

from __future__ import annotations

import logging

from wfomc.algo.core import SolveContext
from wfomc.result import WFOMCResult

from .input import BoundaryProfileInput
from .kernel import evaluate_boundary_profile

logger = logging.getLogger(__name__)


def solve(
    algo_input: BoundaryProfileInput,
    context: SolveContext | None = None,
) -> WFOMCResult:
    """Evaluate every nullary branch and return the undecoded WFOMC value."""

    del context
    arithmetic = algo_input.arithmetic
    result = arithmetic.zero()
    for component_index, component in enumerate(algo_input.components):
        if component.plan is None:
            raise ValueError("Boundary-Profile component is missing its plan")
        component_value, stats = evaluate_boundary_profile(
            component,
            component.plan,
            algo_input.domain_size,
            arithmetic,
        )
        logger.info(
            "Boundary-Profile component solved: component=%d strategy=%s "
            "bp_width=%d join_width=%d estimated_states=%d "
            "estimated_join_pairs=%d materialized_states=%d "
            "join_state_pairs=%d independent_root_states=%d",
            component_index,
            component.plan.strategy,
            component.plan.bp_width,
            component.plan.join_width,
            component.plan.estimated_states,
            component.plan.estimated_join_pairs,
            stats.materialized_states,
            stats.join_state_pairs,
            stats.independent_root_states,
        )
        result = arithmetic.add(
            result,
            arithmetic.multiply(component.graph_weight, component_value),
        )
    return WFOMCResult(result)


__all__ = ["solve"]
