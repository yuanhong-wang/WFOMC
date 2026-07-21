"""Input contract owned by fast-style algorithms."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, TypeAlias

from wfomc.algo.core import AlgoInput, ReducedInputTemplate
from .operations import (
    OptimizedOperations,
    instantiate_optimized_operations,
    materialize_optimized_operations,
)
from wfomc.arithmetic import ArithmeticValue
from wfomc.cell_graph import (
    Cell,
    CellGraphComponent,
    required_profile_predicates,
)
from wfomc.cell_graph.staging import select_cell_graph_structure
from .graph import CellWithEvidenceProfile, build_optimized_cell_graphs
from wfomc.stages import CompiledBranchInstance, CompiledReducedBranch

if TYPE_CHECKING:
    from .graph import _EvidenceOptimizedAnalysis, _OptimizedAnalysis


FastCell: TypeAlias = Cell | CellWithEvidenceProfile


@dataclass(frozen=True)
class OptimizedCellGraphComponent(CellGraphComponent):
    cliques: tuple[tuple[FastCell, ...], ...] = ()
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


@dataclass(frozen=True)
class FastInputTemplate(ReducedInputTemplate):
    """Domain-free weighted cell graphs and static Fast clique layouts."""

    components: tuple[OptimizedCellGraphComponent, ...]

    def instantiate(
        self,
        concrete: CompiledBranchInstance,
    ) -> OptimizedCellGraphInput:
        """Rebind numeric values and create fresh recursive caches for one n."""

        profile = concrete.profile_capacity_constraint
        profile_sizes = (
            None if profile is None else tuple(item.size for item in profile.profiles)
        )
        components = []
        for component in self.components:
            operations = component.weight_operations
            if operations is None:
                raise RuntimeError("Fast input template has no operation tables")
            components.append(
                replace(
                    component,
                    cell_weights=tuple(
                        concrete.arithmetic.coerce(value)
                        for value in component.cell_weights
                    ),
                    pair_weights=tuple(
                        tuple(concrete.arithmetic.coerce(value) for value in row)
                        for row in component.pair_weights
                    ),
                    graph_weight=concrete.arithmetic.coerce(component.graph_weight),
                    weight_operations=instantiate_optimized_operations(
                        operations,
                        arithmetic=concrete.arithmetic,
                        domain_size=len(concrete.domain),
                    ),
                    evidence_profile_sizes=profile_sizes,
                )
            )
        return OptimizedCellGraphInput(
            arithmetic=concrete.arithmetic,
            components=tuple(components),
            domain_size=len(concrete.domain),
        )


def build_input_template(
    compiled: CompiledReducedBranch,
    *,
    input_variant: Hashable,
    modified_cell_symmetry: bool,
) -> FastInputTemplate:
    """Build the expensive Fast cell graphs once without binding n."""

    profile, cell_formulas = select_cell_graph_structure(
        compiled,
        input_variant,
    )
    components = tuple(
        _fast_component(graph, graph_weight, modified_cell_symmetry)
        for graph, graph_weight in build_optimized_cell_graphs(
            compiled.sentence,
            compiled.weights,
            compiled.arithmetic,
            modified_cell_symmetry=modified_cell_symmetry,
            required_unary_preds=required_profile_predicates(profile),
            profile_capacity_constraint=profile,
            cell_formulas=cell_formulas,
        )
    )
    return FastInputTemplate(components)


def _fast_component(
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
    "FastInputTemplate",
    "build_input_template",
]
