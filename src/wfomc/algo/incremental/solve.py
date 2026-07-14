"""Direct incremental WFOMC algorithm over ordered cell-graph inputs."""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

from .input import (
    OrderedCellGraphComponent,
    OrderedCellGraphInput,
)
from wfomc.multinomial import MultinomialCoefficients
from wfomc.result import WFOMCResult

if TYPE_CHECKING:
    from wfomc.engine.runtime import RuntimeContext


def solve(
    algo_input: OrderedCellGraphInput,
    runtime: "RuntimeContext | None" = None,
) -> WFOMCResult:
    if not isinstance(algo_input, OrderedCellGraphInput):
        raise TypeError("incremental algorithm expects an OrderedCellGraphInput")

    MultinomialCoefficients.setup(algo_input.domain_size)
    circle_len = algo_input.circle_len or algo_input.domain_size
    arithmetic = algo_input.arithmetic
    result = arithmetic.zero()
    for component in algo_input.components:
        graph_result = _solve_component(
            component,
            domain_size=algo_input.domain_size,
            predecessor_orders=algo_input.predecessor_orders,
            predecessor_max_order=algo_input.predecessor_max_order,
            has_circular_predecessor=algo_input.has_circular_predecessor,
            circle_len=circle_len,
            arithmetic=arithmetic,
        )
        result += component.graph_weight * graph_result

    return WFOMCResult(result)


def _solve_component(
    component: OrderedCellGraphComponent,
    *,
    domain_size: int,
    predecessor_orders: tuple[int, ...],
    predecessor_max_order: int,
    has_circular_predecessor: bool,
    circle_len: int,
    arithmetic,
) -> object:

    cells = tuple(component.cells)
    weights = _OrderedWeightTables(component)
    n_cells = len(cells)

    def step_weight(
        active_cells, cell, j, ivec, last_cells, first_cell, w_old, cur_idx
    ):
        old_ivec = list(ivec)
        w_new = w_old * weights.cell(cell)
        if cur_idx == circle_len - 2 and first_cell is not None:
            w_new *= weights.predecessor(first_cell, cell, 1)
            old_ivec[active_cells.index(first_cell)] -= 1
        if last_cells is not None:
            for pred_idx in predecessor_orders:
                if cur_idx >= pred_idx - 1:
                    pred_cell = last_cells[-pred_idx]
                    w_new *= weights.predecessor(cell, pred_cell, pred_idx)
                    old_ivec[active_cells.index(pred_cell)] -= 1
            new_last_cells = last_cells[1:] + (cell,)
        else:
            new_last_cells = None
        w_new *= reduce(
            lambda left, right: left * right,
            (
                weights.pair(cell, other_cell) ** old_ivec[k]
                for k, other_cell in enumerate(active_cells)
            ),
            arithmetic.one(),
        )
        new_ivec = tuple(num if k != j else num + 1 for k, num in enumerate(ivec))
        return (
            w_new,
            new_ivec,
            tuple(new_last_cells) if new_last_cells is not None else None,
        )

    def init_state(cell):
        return (
            None
            if not predecessor_orders
            else tuple(cell for _ in range(predecessor_max_order)),
            cell if has_circular_predecessor else None,
        )

    allocation = component.cell_evidence_allocation
    if allocation is not None:
        return _threaded_graph(
            component,
            cells,
            allocation,
            domain_size,
            step_weight,
            init_state,
            arithmetic,
        )

    table = {}
    for i, cell in enumerate(cells):
        last_cells, first_cell = init_state(cell)
        table[(tuple(int(k == i) for k in range(n_cells)), last_cells, first_cell)] = (
            weights.cell(cell)
        )

    for cur_idx in range(domain_size - 1):
        old_table = table
        table = {}
        for j, cell in enumerate(cells):
            for (ivec, last_cells, first_cell), w_old in old_table.items():
                w_new, new_ivec, new_last_cells = step_weight(
                    cells,
                    cell,
                    j,
                    ivec,
                    last_cells,
                    first_cell,
                    w_old,
                    cur_idx,
                )
                key = (new_ivec, new_last_cells, first_cell)
                table[key] = table.get(key, arithmetic.zero()) + w_new

    graph_result = arithmetic.zero()
    for (_ivec, _last_cells, _first_cell), weight in table.items():
        graph_result += weight
    return graph_result


def _threaded_graph(
    component: OrderedCellGraphComponent,
    cells,
    allocation,
    domain_size,
    step_weight,
    init_state,
    arithmetic,
) -> object:

    weights = _OrderedWeightTables(component)
    active_indices = allocation.compatible_cell_indices
    active_cells = [cells[idx] for idx in active_indices]
    n_cells = len(active_cells)
    initial_remaining = allocation.initial_remaining_counts()

    table = {}
    for i, (cell, original_idx) in enumerate(zip(active_cells, active_indices)):
        last_cells, first_cell = init_state(cell)
        ivec0 = tuple(int(k == i) for k in range(n_cells))
        w0 = weights.cell(cell)
        for rem in allocation.next_remaining_counts(initial_remaining, original_idx):
            key = (ivec0, last_cells, first_cell, rem)
            table[key] = table.get(key, arithmetic.zero()) + w0

    for cur_idx in range(domain_size - 1):
        old_table = table
        table = {}
        for j, (cell, original_idx) in enumerate(zip(active_cells, active_indices)):
            for (ivec, last_cells, first_cell, old_rem), w_old in old_table.items():
                new_remaining_states = allocation.next_remaining_counts(
                    old_rem,
                    original_idx,
                )
                if not new_remaining_states:
                    continue
                w_new, new_ivec, new_last_cells = step_weight(
                    active_cells,
                    cell,
                    j,
                    ivec,
                    last_cells,
                    first_cell,
                    w_old,
                    cur_idx,
                )
                for new_rem in new_remaining_states:
                    key = (new_ivec, new_last_cells, first_cell, new_rem)
                    table[key] = table.get(key, arithmetic.zero()) + w_new

    zero_rem = tuple(0 for _ in allocation.evidence_profile_sizes)
    graph_result = arithmetic.zero()
    for (_ivec, _last_cells, _first_cell, rem), weight in table.items():
        if rem == zero_rem:
            graph_result += weight
    return allocation.normalize_ordered_evidence_assignments(
        graph_result,
        arithmetic,
    )


class _OrderedWeightTables:
    def __init__(self, component: OrderedCellGraphComponent):
        self.component = component
        self.cell_index = {cell: idx for idx, cell in enumerate(component.cells)}

    def cell(self, cell: object) -> object:
        return self.component.cell_weights[self.cell_index[cell]]

    def pair(self, left: object, right: object) -> object:
        return self.component.pair_weights[self.cell_index[left]][
            self.cell_index[right]
        ]

    def predecessor(self, left: object, right: object, order: int) -> object:
        predk_tables = self.component.predk_pair_tables or {}
        table = predk_tables.get(order)
        if table is None and order == 1:
            table = self.component.circular_predecessor_pair_tables
        if table is None:
            raise RuntimeError(
                f"ordered component is missing predecessor pair table for PRED{order}"
            )
        return table[self.cell_index[left]][self.cell_index[right]]


__all__ = ["solve"]
