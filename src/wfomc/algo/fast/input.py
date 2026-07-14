"""Input contract owned by fast-style algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from wfomc.algo.core import AlgoInput, AlgoOptions
from .operations import (
    OptimizedOperations,
    materialize_optimized_operations,
)
from wfomc.arithmetic import ArithmeticValue
from wfomc.cell_graph import (
    Cell,
    CellGraphComponent,
    profile_cell_formulas,
    required_profile_predicates,
)
from .graph import CellWithEvidenceProfile, build_optimized_cell_graphs
from wfomc.problem import CompiledProblem

if TYPE_CHECKING:
    from .graph import _EvidenceOptimizedAnalysis, _OptimizedAnalysis


FastCell: TypeAlias = Cell | CellWithEvidenceProfile


@dataclass(frozen=True)
class OptimizedCellGraphComponent(CellGraphComponent):
    cliques: tuple[tuple[FastCell, ...], ...] = ()
    i1_independent: tuple[int, ...] = ()
    i2_independent: tuple[int, ...] = ()
    non_independent: tuple[int, ...] = ()
    non_independent_index: Mapping[int, int] = field(default_factory=dict)
    modified_cell_symmetry: bool = False
    weight_operations: OptimizedOperations | None = None
    evidence_profile_sizes: tuple[int, ...] | None = None
    evidence_profile_cliques: Mapping[int, tuple[int, ...]] | None = None


@dataclass(frozen=True)
class OptimizedCellGraphInput(AlgoInput):
    components: tuple[OptimizedCellGraphComponent, ...] = ()
    domain_size: int = 0


def build_input(
    reduced: CompiledProblem,
    *,
    domain_size: int,
    modified_cell_symmetry: bool,
    options: AlgoOptions,
) -> OptimizedCellGraphInput:
    components = tuple(
        _fast_component(reduced, graph, graph_weight, modified_cell_symmetry)
        for graph, graph_weight in build_optimized_cell_graphs(
            reduced.sentence,
            reduced.weights,
            reduced.arithmetic,
            domain_size=domain_size,
            modified_cell_symmetry=modified_cell_symmetry,
            required_unary_preds=required_profile_predicates(
                reduced.profile_capacity_constraint
            ),
            profile_capacity_constraint=reduced.profile_capacity_constraint,
            cell_formulas=profile_cell_formulas(reduced.profile_capacity_constraint),
        )
    )
    return OptimizedCellGraphInput(
        algo=None,
        options=options,
        arithmetic=reduced.arithmetic,
        components=components,
        domain_size=domain_size,
    )


def _fast_component(
    reduced: CompiledProblem,
    graph: "_OptimizedAnalysis | _EvidenceOptimizedAnalysis",
    graph_weight: ArithmeticValue,
    modified_cell_symmetry: bool,
) -> OptimizedCellGraphComponent:
    cells = tuple(graph.get_cells())
    cell_weights, pair_weights = graph.get_all_weights()
    return OptimizedCellGraphComponent(
        cells=cells,
        cell_weights=tuple(cell_weights),
        pair_weights=tuple(tuple(row) for row in pair_weights),
        graph_weight=graph_weight,
        cliques=tuple(tuple(clique) for clique in graph.cliques),
        i1_independent=tuple(graph.i1_ind),
        i2_independent=tuple(graph.i2_ind),
        non_independent=tuple(graph.nonind),
        non_independent_index=dict(graph.nonind_map),
        modified_cell_symmetry=modified_cell_symmetry,
        weight_operations=materialize_optimized_operations(graph),
        evidence_profile_sizes=(
            None
            if graph.evidence_profile_sizes is None
            else tuple(graph.evidence_profile_sizes)
        ),
        evidence_profile_cliques=(
            None
            if graph.evidence_profile_cliques is None
            else {
                profile: tuple(cliques)
                for profile, cliques in graph.evidence_profile_cliques.items()
            }
        ),
    )


__all__ = [
    "OptimizedCellGraphComponent",
    "OptimizedCellGraphInput",
    "build_input",
]
