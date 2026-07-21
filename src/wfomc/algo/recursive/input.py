"""Input preparation for the recursive algorithm."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from wfomc.arithmetic import ArithmeticValue
from wfomc.algo.core import AlgoInput, ReducedInputTemplate
from wfomc.cell_graph import CellGraphComponent, CellGraphData, build_cell_graphs
from wfomc.cell_graph.staging import (
    rebind_component,
    select_cell_graph_structure,
)
from wfomc.stages import (
    CompiledBranchInstance,
    CompiledReducedBranch,
)

@dataclass(frozen=True)
class RecursiveInput(AlgoInput):
    components: tuple[CellGraphComponent, ...] = ()
    domain_size: int = 0


@dataclass(frozen=True)
class RecursiveInputTemplate(ReducedInputTemplate):
    """Reusable ordered cell graphs for recursive solving."""

    components: tuple[CellGraphComponent, ...]

    def instantiate(
        self,
        concrete: CompiledBranchInstance,
    ) -> RecursiveInput:
        """Bind recursive graph weights for one concrete domain."""

        return RecursiveInput(
            arithmetic=concrete.arithmetic,
            components=tuple(
                rebind_component(component, concrete, include_evidence=False)
                for component in self.components
            ),
            domain_size=len(concrete.domain),
        )


def build_input_template(
    compiled: CompiledReducedBranch,
    *,
    input_variant: Hashable,
) -> RecursiveInputTemplate:
    """Build recursive cell graphs without binding a domain."""

    _profile, cell_formulas = select_cell_graph_structure(compiled, input_variant)
    components = tuple(
        _component(data, graph_weight)
        for data, graph_weight in build_cell_graphs(
            compiled.sentence,
            compiled.weights,
            compiled.arithmetic,
            leq_pred=compiled.feature_set.leq_predicate,
            cell_formulas=cell_formulas,
        )
    )
    return RecursiveInputTemplate(components)


def _component(
    data: CellGraphData,
    graph_weight: ArithmeticValue,
) -> CellGraphComponent:
    return CellGraphComponent(
        cells=data.cells,
        cell_weights=data.cell_weights,
        pair_weights=data.pair_weights(),
        graph_weight=graph_weight,
    )


__all__ = [
    "RecursiveInput",
    "RecursiveInputTemplate",
    "build_input_template",
]
