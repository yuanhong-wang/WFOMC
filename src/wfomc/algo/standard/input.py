"""Standard WFOMC input contract and materialization."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wfomc.arithmetic import ArithmeticValue
from wfomc.algo.core import AlgoInput, ReducedInputTemplate
from wfomc.cell_graph import (
    CellGraphComponent,
    CellGraphData,
    build_cell_graphs,
    materialize_cell_evidence,
    required_profile_predicates,
)
from wfomc.cell_graph.staging import (
    rebind_component,
    select_cell_graph_structure,
)
from wfomc.stages import CompiledBranchInstance, CompiledReducedBranch

if TYPE_CHECKING:
    from wfomc.evidence.profile import ProfileCapacityConstraint


@dataclass(frozen=True)
class StandardInput(AlgoInput):
    components: tuple[CellGraphComponent, ...]
    domain_size: int


@dataclass(frozen=True)
class StandardInputTemplate(ReducedInputTemplate):
    """Reusable scalar cell graphs for the Standard algorithm."""

    components: tuple[CellGraphComponent, ...]

    def instantiate(
        self,
        concrete: CompiledBranchInstance,
    ) -> StandardInput:
        """Bind arithmetic and unary-evidence capacities for one domain."""

        return StandardInput(
            arithmetic=concrete.arithmetic,
            components=tuple(
                rebind_component(component, concrete, include_evidence=True)
                for component in self.components
            ),
            domain_size=len(concrete.domain),
        )


def build_input_template(
    compiled: CompiledReducedBranch,
    *,
    input_variant: Hashable,
) -> StandardInputTemplate:
    """Build Standard cell graphs without binding a concrete domain."""

    profile, cell_formulas = select_cell_graph_structure(compiled, input_variant)
    components = tuple(
        _standard_component_from_cell_graph(
            profile,
            cell_graph,
            graph_weight,
        )
        for cell_graph, graph_weight in build_cell_graphs(
            compiled.sentence,
            compiled.weights,
            compiled.arithmetic,
            required_unary_preds=required_profile_predicates(profile),
            cell_formulas=cell_formulas,
        )
    )
    return StandardInputTemplate(components)


def _standard_component_from_cell_graph(
    profile: "ProfileCapacityConstraint | None",
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
            profile,
            cell_graph.cells,
        ),
    )


__all__ = [
    "StandardInput",
    "StandardInputTemplate",
    "build_input_template",
]
