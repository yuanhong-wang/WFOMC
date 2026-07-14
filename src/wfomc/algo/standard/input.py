"""Standard WFOMC input contract and materialization."""

from __future__ import annotations

from dataclasses import dataclass

from wfomc.arithmetic import ArithmeticValue
from wfomc.algo.core import AlgoInput, AlgoOptions
from wfomc.cell_graph import (
    CellGraphComponent,
    CellGraphData,
    build_cell_graphs,
    materialize_cell_evidence,
    profile_cell_formulas,
    required_profile_predicates,
)
from wfomc.problem import CompiledProblem


@dataclass(frozen=True)
class StandardInput(AlgoInput):
    components: tuple[CellGraphComponent, ...]
    domain_size: int


def build_input(
    reduced: CompiledProblem,
    *,
    options: AlgoOptions,
) -> StandardInput:
    components = tuple(
        _standard_component_from_cell_graph(reduced, cell_graph, graph_weight)
        for cell_graph, graph_weight in build_cell_graphs(
            reduced.sentence,
            reduced.weights,
            reduced.arithmetic,
            required_unary_preds=required_profile_predicates(
                reduced.profile_capacity_constraint
            ),
            cell_formulas=profile_cell_formulas(reduced.profile_capacity_constraint),
        )
    )
    return StandardInput(
        algo=None,
        options=options,
        arithmetic=reduced.arithmetic,
        components=components,
        domain_size=len(reduced.domain),
    )


def _standard_component_from_cell_graph(
    reduced: CompiledProblem,
    cell_graph: CellGraphData,
    graph_weight: ArithmeticValue,
) -> CellGraphComponent:
    pair_weights = cell_graph.pair_weights()
    return CellGraphComponent(
        cells=cell_graph.cells,
        cell_weights=cell_graph.cell_weights,
        pair_weights=pair_weights,
        graph_weight=graph_weight,
        cell_evidence_allocation=materialize_cell_evidence(
            reduced.profile_capacity_constraint,
            cell_graph.cells,
        ),
    )


__all__ = [
    "StandardInput",
    "build_input",
]
