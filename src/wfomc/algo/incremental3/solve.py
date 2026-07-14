"""Direct incremental3 WFOMC algorithm over counting-DP inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from .input import (
    CountingCellGraphComponent,
    CountingDPInput,
)
from wfomc.multinomial import MultinomialCoefficients
from wfomc.result import WFOMCResult

if TYPE_CHECKING:
    from wfomc.engine.runtime import RuntimeContext


def solve(
    algo_input: CountingDPInput,
    runtime: "RuntimeContext | None" = None,
) -> WFOMCResult:
    if not isinstance(algo_input, CountingDPInput):
        raise TypeError("incremental3 algorithm expects a CountingDPInput")
    if algo_input.counting_state is None:
        raise RuntimeError("incremental3 algorithm requires counting_state")
    if algo_input.unary_cardinality_masks is None:
        raise RuntimeError("incremental3 algorithm requires unary cardinality masks")

    MultinomialCoefficients.setup(algo_input.domain_size)
    arithmetic = algo_input.arithmetic
    result = arithmetic.zero()
    for component in algo_input.components:
        result += _solve_component(
            component,
            domain_size=algo_input.domain_size,
            counting_state=algo_input.counting_state,
            unary_masks=algo_input.unary_cardinality_masks,
            has_linear_order=algo_input.has_linear_order,
            arithmetic=arithmetic,
        )

    return WFOMCResult(result)


def _solve_component(
    component: CountingCellGraphComponent,
    *,
    domain_size: int,
    counting_state: object,
    unary_masks: object,
    has_linear_order: bool,
    arithmetic,
) -> object:
    from .counting_kernel import (
        ConfigSpace,
        _make_domain_recursion,
        build_t_update_dict,
    )
    from wfomc.cell_graph import (
        CellConfigCoefficientBasis,
        CellEvidenceAllocation,
    )

    cells = tuple(component.cells)
    n_cells = len(cells)

    if not (
        component.counting_initial_states
        and component.counting_binary_relation_weights is not None
    ):
        raise RuntimeError("counting cell-graph weight tables are not materialized")
    w2t, w, r = _build_weight_from_materialized_tables(component, arithmetic)
    unary_mask = unary_masks.build_mask(cells)
    t_update_dict = build_t_update_dict(
        r,
        n_cells,
        counting_state,
        arithmetic,
    )
    space = ConfigSpace((n_cells,) + tuple(counting_state.c_type_shape))
    domain_recursion = _make_domain_recursion(
        t_update_dict,
        space,
        counting_state,
        has_linear_order,
        arithmetic,
    )

    allocation = component.cell_evidence_allocation
    if allocation is None:
        allocation = CellEvidenceAllocation.unconstrained(n_cells, domain_size)
    coefficient_basis = (
        CellConfigCoefficientBasis.RELATIVE_TO_CELL_MULTINOMIAL
        if has_linear_order
        else CellConfigCoefficientBasis.ABSOLUTE
    )

    subtotal = arithmetic.zero()
    for config, coefficient in allocation.iter_config_coefficients(
        arithmetic,
        coefficient_basis,
    ):
        if any(unary_masks.check(config, unary_mask)):
            continue

        init_list = list(space.zero)
        weight = arithmetic.one()
        for idx, count in enumerate(config):
            init_state = (idx,) + w2t[idx]
            init_list[space.offset(init_state)] = count
            weight *= w[idx] ** count

        subtotal += coefficient * weight * domain_recursion(tuple(init_list))

    return component.graph_weight * subtotal


def _build_weight_from_materialized_tables(
    component: CountingCellGraphComponent,
    arithmetic,
) -> tuple[dict[int, tuple[int, ...]], object, object]:
    from collections import defaultdict

    n_cells = len(component.cells)
    if len(component.counting_initial_states) != n_cells:
        raise RuntimeError(
            "counting component initial-state table does not match cell count"
        )
    if len(component.counting_binary_relation_weights) != n_cells:
        raise RuntimeError(
            "counting component binary relation table does not match cell count"
        )

    w2t = {
        idx: tuple(initial_state)
        for idx, initial_state in enumerate(component.counting_initial_states)
    }
    w = defaultdict(arithmetic.zero)
    r = defaultdict(lambda: defaultdict(arithmetic.zero))

    for idx, weight in enumerate(component.cell_weights):
        w[idx] += weight

    for left_idx, row in enumerate(component.counting_binary_relation_weights):
        if len(row) != n_cells:
            raise RuntimeError(
                "counting component binary relation row does not match cell count"
            )
        for right_idx, entries in enumerate(row):
            for forward_delta, reverse_delta, weight in entries:
                r[(left_idx, right_idx)][
                    (tuple(forward_delta), tuple(reverse_delta))
                ] += weight

    return w2t, w, r


__all__ = ["solve"]
