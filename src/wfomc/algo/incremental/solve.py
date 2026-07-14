"""Direct incremental WFOMC algorithm over ordered cell-graph inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .input import (
    OrderedCellGraphComponent,
    OrderedCellGraphInput,
)
from wfomc.multinomial import MultinomialCoefficients
from wfomc.result import WFOMCResult

if TYPE_CHECKING:
    from wfomc.engine.runtime import RuntimeContext


_ZERO_FACTOR = object()
_ONE_FACTOR = object()
_MISSING = object()


@dataclass(frozen=True)
class _PairRow:
    zero_indices: tuple[int, ...]
    weighted_factors: tuple[tuple[int, object], ...]


def solve(
    algo_input: OrderedCellGraphInput,
    runtime: "RuntimeContext | None" = None,
) -> WFOMCResult:
    if not isinstance(algo_input, OrderedCellGraphInput):
        raise TypeError("incremental algorithm expects an OrderedCellGraphInput")

    MultinomialCoefficients.setup(algo_input.domain_size)
    circle_len = (
        algo_input.circle_len
        if algo_input.circle_len is not None
        else algo_input.domain_size
    )
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
        result = arithmetic.add(
            result,
            arithmetic.multiply(component.graph_weight, graph_result),
        )

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

    allocation = component.cell_evidence_allocation
    active_indices = (
        allocation.compatible_cell_indices
        if allocation is not None
        else tuple(range(len(component.cells)))
    )
    weights = _OrderedWeightTables(component, active_indices, arithmetic)
    n_cells = len(active_indices)

    def step_weight(j, ivec, last_cells, first_cell, w_old, cur_idx):
        old_ivec = list(ivec)
        predecessor_factors = []
        if cur_idx == circle_len - 2 and first_cell is not None:
            factor = weights.predecessor(first_cell, j, 1)
            if factor is _ZERO_FACTOR:
                return None
            if factor is not _ONE_FACTOR:
                predecessor_factors.append(factor)
            old_ivec[first_cell] -= 1
        if last_cells is not None:
            for order in predecessor_orders:
                if cur_idx >= order - 1:
                    predecessor_cell = last_cells[-order]
                    factor = weights.predecessor(j, predecessor_cell, order)
                    if factor is _ZERO_FACTOR:
                        return None
                    if factor is not _ONE_FACTOR:
                        predecessor_factors.append(factor)
                    old_ivec[predecessor_cell] -= 1
            new_last_cells = last_cells[1:] + (j,)
        else:
            new_last_cells = None

        cell_factor = weights.cell_factor(j)
        if cell_factor is _ZERO_FACTOR:
            return None
        w_new = w_old
        if cell_factor is not _ONE_FACTOR:
            w_new = arithmetic.multiply(w_new, cell_factor)
        for factor in predecessor_factors:
            w_new = arithmetic.multiply(w_new, factor)

        pair_row = weights.pair_row(j)
        if any(old_ivec[index] > 0 for index in pair_row.zero_indices):
            return None
        for index, factor in pair_row.weighted_factors:
            exponent = old_ivec[index]
            if exponent:
                w_new = arithmetic.multiply(
                    w_new,
                    arithmetic.power(factor, exponent),
                )

        new_ivec = tuple(num if k != j else num + 1 for k, num in enumerate(ivec))
        return (
            w_new,
            new_ivec,
            new_last_cells,
        )

    def init_state(cell_index):
        return (
            None
            if not predecessor_orders
            else tuple(cell_index for _ in range(predecessor_max_order)),
            cell_index if has_circular_predecessor else None,
        )

    if allocation is not None:
        return _threaded_graph(
            allocation,
            active_indices,
            domain_size,
            step_weight,
            init_state,
            weights,
            arithmetic,
        )

    table = {}
    for i in range(n_cells):
        if weights.cell_factor(i) is _ZERO_FACTOR:
            continue
        last_cells, first_cell = init_state(i)
        table[(tuple(int(k == i) for k in range(n_cells)), last_cells, first_cell)] = (
            weights.cell(i)
        )

    for cur_idx in range(domain_size - 1):
        old_table = table
        table = {}
        for j in range(n_cells):
            for (ivec, last_cells, first_cell), w_old in old_table.items():
                step = step_weight(
                    j,
                    ivec,
                    last_cells,
                    first_cell,
                    w_old,
                    cur_idx,
                )
                if step is None:
                    continue
                w_new, new_ivec, new_last_cells = step
                key = (new_ivec, new_last_cells, first_cell)
                _accumulate(table, key, w_new, arithmetic)

    graph_result = arithmetic.zero()
    for (_ivec, _last_cells, _first_cell), weight in table.items():
        graph_result = arithmetic.add(graph_result, weight)
    return graph_result


def _threaded_graph(
    allocation,
    active_indices,
    domain_size,
    step_weight,
    init_state,
    weights,
    arithmetic,
) -> object:

    n_cells = len(active_indices)
    initial_remaining = allocation.initial_remaining_counts()

    table = {}
    for i, original_idx in enumerate(active_indices):
        if weights.cell_factor(i) is _ZERO_FACTOR:
            continue
        last_cells, first_cell = init_state(i)
        ivec0 = tuple(int(k == i) for k in range(n_cells))
        w0 = weights.cell(i)
        for rem in allocation.next_remaining_counts(initial_remaining, original_idx):
            key = (ivec0, last_cells, first_cell, rem)
            _accumulate(table, key, w0, arithmetic)

    for cur_idx in range(domain_size - 1):
        old_table = table
        table = {}
        for j, original_idx in enumerate(active_indices):
            for (ivec, last_cells, first_cell, old_rem), w_old in old_table.items():
                new_remaining_states = allocation.next_remaining_counts(
                    old_rem,
                    original_idx,
                )
                if not new_remaining_states:
                    continue
                step = step_weight(
                    j,
                    ivec,
                    last_cells,
                    first_cell,
                    w_old,
                    cur_idx,
                )
                if step is None:
                    continue
                w_new, new_ivec, new_last_cells = step
                for new_rem in new_remaining_states:
                    key = (new_ivec, new_last_cells, first_cell, new_rem)
                    _accumulate(table, key, w_new, arithmetic)

    zero_rem = tuple(0 for _ in allocation.evidence_profile_sizes)
    graph_result = arithmetic.zero()
    for (_ivec, _last_cells, _first_cell, rem), weight in table.items():
        if rem == zero_rem:
            graph_result = arithmetic.add(graph_result, weight)
    return allocation.normalize_ordered_evidence_assignments(
        graph_result,
        arithmetic,
    )


class _OrderedWeightTables:
    """Active-cell-local weights with zero/one factors classified once."""

    def __init__(self, component, active_indices, arithmetic):
        self.cell_weights = tuple(
            component.cell_weights[index] for index in active_indices
        )
        self.cell_factors = tuple(
            _classify_factor(value, arithmetic) for value in self.cell_weights
        )
        self.pair_rows = tuple(
            _compile_pair_row(
                component.pair_weights[original_left],
                active_indices,
                arithmetic,
            )
            for original_left in active_indices
        )
        self.predecessor_tables = {
            order: _compile_factor_matrix(table, active_indices, arithmetic)
            for order, table in (component.predk_pair_tables or {}).items()
        }
        self.circular_predecessor_table = (
            None
            if component.circular_predecessor_pair_tables is None
            else _compile_factor_matrix(
                component.circular_predecessor_pair_tables,
                active_indices,
                arithmetic,
            )
        )

    def cell(self, index: int) -> object:
        return self.cell_weights[index]

    def cell_factor(self, index: int) -> object:
        return self.cell_factors[index]

    def pair_row(self, index: int) -> _PairRow:
        return self.pair_rows[index]

    def predecessor(self, left: int, right: int, order: int) -> object:
        table = self.predecessor_tables.get(order)
        if table is None and order == 1:
            table = self.circular_predecessor_table
        if table is None:
            raise RuntimeError(
                f"ordered component is missing predecessor pair table for PRED{order}"
            )
        return table[left][right]


def _classify_factor(value, arithmetic):
    if arithmetic.is_zero(value):
        return _ZERO_FACTOR
    if arithmetic.is_one(value):
        return _ONE_FACTOR
    return value


def _compile_pair_row(row, active_indices, arithmetic) -> _PairRow:
    zero_indices = []
    weighted_factors = []
    for local_index, original_index in enumerate(active_indices):
        factor = _classify_factor(row[original_index], arithmetic)
        if factor is _ZERO_FACTOR:
            zero_indices.append(local_index)
        elif factor is not _ONE_FACTOR:
            weighted_factors.append((local_index, factor))
    return _PairRow(tuple(zero_indices), tuple(weighted_factors))


def _compile_factor_matrix(matrix, active_indices, arithmetic):
    return tuple(
        tuple(
            _classify_factor(matrix[original_left][original_right], arithmetic)
            for original_right in active_indices
        )
        for original_left in active_indices
    )


def _accumulate(table, key, value, arithmetic) -> None:
    current = table.get(key, _MISSING)
    table[key] = (
        value if current is _MISSING else arithmetic.add(current, value)
    )


__all__ = ["solve"]
