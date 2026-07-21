"""Input contract owned by the incremental3 algorithm."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from wfomc.arithmetic import ArithmeticValue
from wfomc.algo.core import AlgoInput, ReducedInputTemplate
from wfomc.fol import Predicate
from .counting_state import CountingState, UnaryCardinalityMasks
from wfomc.cell_graph import (
    Cell,
    CellGraphComponent,
    CellGraphData,
    build_cell_graphs,
    materialize_cell_evidence,
    required_profile_predicates,
)
from wfomc.cell_graph.staging import (
    CellGraphInputVariant,
    rebind_component,
    select_cell_graph_structure,
)
from wfomc.stages import (
    CompiledBranchInstance,
    CompiledReducedBranch,
)

if TYPE_CHECKING:
    from wfomc.arithmetic import ArithmeticContext
    from wfomc.evidence.profile import ProfileCapacityConstraint


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


@dataclass(frozen=True)
class Incremental3InputVariant:
    """Domain-selected evidence and native counting-state structure."""

    evidence: CellGraphInputVariant
    counting_state: CountingState


@dataclass(frozen=True)
class Incremental3InputTemplate(ReducedInputTemplate):
    """Reusable cell graphs for one native counting-state structure."""

    components: tuple[CountingCellGraphComponent, ...]
    counting_state: CountingState
    unary_cardinality_masks: UnaryCardinalityMasks
    has_linear_order: bool

    def instantiate(
        self,
        concrete: CompiledBranchInstance,
    ) -> CountingDPInput:
        """Bind graph arithmetic and evidence capacities for one domain."""

        components = []
        for component in self.components:
            rebound = rebind_component(component, concrete, include_evidence=True)
            relation_weights = component.counting_binary_relation_weights
            components.append(
                replace(
                    rebound,
                    counting_binary_relation_weights=(
                        None
                        if relation_weights is None
                        else tuple(
                            tuple(
                                tuple(
                                    (
                                        forward_delta,
                                        reverse_delta,
                                        concrete.arithmetic.coerce(weight),
                                    )
                                    for forward_delta, reverse_delta, weight in entries
                                )
                                for entries in row
                            )
                            for row in relation_weights
                        )
                    ),
                )
            )
        return CountingDPInput(
            arithmetic=concrete.arithmetic,
            components=tuple(components),
            domain_size=len(concrete.domain),
            counting_state=self.counting_state,
            unary_cardinality_masks=self.unary_cardinality_masks,
            has_linear_order=self.has_linear_order,
        )


def build_input_template(
    compiled: CompiledReducedBranch,
    *,
    input_variant: Incremental3InputVariant,
) -> Incremental3InputTemplate:
    """Build one reusable graph for a selected native counting automaton."""

    from .counting_state import build_counting_state_for_normal_form

    profile, cell_formulas = select_cell_graph_structure(
        compiled,
        input_variant.evidence,
    )
    _general_state, unary_masks = build_counting_state_for_normal_form(
        compiled.reduced_problem.normal_form,
    )
    state = input_variant.counting_state
    leq = compiled.feature_set.leq_predicate
    required_unary_preds = (
        required_profile_predicates(profile)
        | unary_masks.required_predicates()
    )
    components = tuple(
        _counting_component(
            compiled.arithmetic,
            profile,
            data,
            graph_weight,
            state,
        )
        for data, graph_weight in build_cell_graphs(
            compiled.sentence,
            compiled.weights,
            compiled.arithmetic,
            leq_pred=leq,
            required_unary_preds=required_unary_preds,
            cell_formulas=cell_formulas,
            projected_binary_preds=state.projected_predicates,
        )
    )
    return Incremental3InputTemplate(
        components=components,
        counting_state=state,
        unary_cardinality_masks=unary_masks,
        has_linear_order=leq is not None,
    )


def _counting_component(
    arithmetic: "ArithmeticContext",
    profile: "ProfileCapacityConstraint | None",
    data: CellGraphData,
    graph_weight: ArithmeticValue,
    state: CountingState,
) -> CountingCellGraphComponent:
    initial_states = tuple(_initial_state(cell, state) for cell in data.cells)
    accepting_states = tuple(
        tuple(counter.accepting_states_for_cell(cell) for counter in state.row_counters)
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
                if arithmetic.is_zero(weight):
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
            profile,
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
    "Incremental3InputTemplate",
    "Incremental3InputVariant",
    "build_input_template",
]
