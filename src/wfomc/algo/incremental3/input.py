"""Input contract owned by the incremental3 algorithm."""

from __future__ import annotations

from dataclasses import dataclass

from wfomc.arithmetic import ArithmeticValue
from wfomc.algo.core import AlgoInput, AlgoOptions
from wfomc.fol import Predicate
from .counting_state import CountingState, UnaryCardinalityMasks
from wfomc.cell_graph import (
    Cell,
    CellGraphComponent,
    CellGraphData,
    build_cell_graphs,
    materialize_cell_evidence,
    profile_cell_formulas,
    required_profile_predicates,
)
from wfomc.problem import CompiledProblem
from wfomc.engine.features import FeatureSet


@dataclass(frozen=True)
class CountingCellGraphComponent(CellGraphComponent):
    counting_initial_states: tuple[tuple[int, ...] | None, ...] = ()
    counting_accepting_states: tuple[tuple[frozenset[int], ...], ...] = ()
    nullary_assignments: tuple[tuple[Predicate, bool], ...] = ()
    counting_binary_relation_weights: (
        tuple[tuple[tuple[tuple[int, ...], tuple[int, ...], ArithmeticValue], ...], ...]
        | None
    ) = None


@dataclass(frozen=True)
class CountingDPInput(AlgoInput):
    components: tuple[CountingCellGraphComponent, ...] = ()
    domain_size: int = 0
    counting_state: CountingState | None = None
    unary_cardinality_masks: UnaryCardinalityMasks | None = None
    has_linear_order: bool = False


def build_input(
    reduced: CompiledProblem,
    *,
    counting_state: CountingState,
    unary_cardinality_masks: UnaryCardinalityMasks,
    options: AlgoOptions,
    features: FeatureSet,
) -> CountingDPInput:
    leq = features.leq_predicate
    required_unary_preds = required_profile_predicates(
        reduced.profile_capacity_constraint
    ) | unary_cardinality_masks.required_predicates()
    components = tuple(
        _counting_component(reduced, data, graph_weight, counting_state)
        for data, graph_weight in build_cell_graphs(
            reduced.sentence,
            reduced.weights,
            reduced.arithmetic,
            leq_pred=leq,
            required_unary_preds=required_unary_preds,
            cell_formulas=profile_cell_formulas(reduced.profile_capacity_constraint),
            projected_binary_preds=counting_state.projected_predicates,
        )
    )
    return CountingDPInput(
        algo=None,
        options=options,
        arithmetic=reduced.arithmetic,
        components=components,
        domain_size=len(reduced.domain),
        counting_state=counting_state,
        unary_cardinality_masks=unary_cardinality_masks,
        has_linear_order=leq is not None,
    )


def _counting_component(
    reduced: CompiledProblem,
    data: CellGraphData,
    graph_weight: ArithmeticValue,
    state: CountingState,
) -> CountingCellGraphComponent:
    initial_states = tuple(_initial_state(cell, state) for cell in data.cells)
    accepting_states = tuple(
        tuple(
            counter.accepting_states_for_cell(cell)
            for counter in state.row_counters
        )
        for cell in data.cells
    )
    projected_preds = state.projected_predicates
    counter_indices = state.counter_projection_indices
    relation_rows = []
    for left_idx in range(len(data.cells)):
        row = []
        for right_idx in range(len(data.cells)):
            entries = []
            factor = data.pair_factors[left_idx][right_idx]
            projected_weights = factor.counting_weights or (
                ((0, factor.total_weight),) if not projected_preds else ()
            )
            for evidence_idx, weight in projected_weights:
                if reduced.arithmetic.is_zero(weight):
                    continue
                reverse_projected = tuple(
                    int(bool((evidence_idx >> (2 * pred_idx)) & 1))
                    for pred_idx in range(len(projected_preds))
                )
                forward_projected = tuple(
                    int(bool((evidence_idx >> (2 * pred_idx + 1)) & 1))
                    for pred_idx in range(len(projected_preds))
                )
                reverse_delta = tuple(
                    reverse_projected[index] for index in counter_indices
                )
                forward_delta = tuple(
                    forward_projected[index] for index in counter_indices
                )
                entries.append((forward_delta, reverse_delta, weight))
            row.append(tuple(entries))
        relation_rows.append(tuple(row))
    return CountingCellGraphComponent(
        cells=data.cells,
        cell_weights=data.cell_weights,
        pair_weights=data.pair_weights(),
        graph_weight=graph_weight,
        cell_evidence_allocation=materialize_cell_evidence(
            reduced.profile_capacity_constraint,
            data.cells,
        ),
        counting_initial_states=initial_states,
        counting_accepting_states=accepting_states,
        counting_binary_relation_weights=tuple(relation_rows),
        nullary_assignments=data.nullary_assignments,
    )


def _initial_state(cell: Cell, state: CountingState) -> tuple[int, ...] | None:
    slots = []
    for counter in state.row_counters:
        initial = counter.initial_state(cell.is_positive(counter.predicate))
        if initial is None:
            return None
        slots.append(initial)
    return tuple(slots)


__all__ = [
    "CountingCellGraphComponent",
    "CountingDPInput",
    "build_input",
]
