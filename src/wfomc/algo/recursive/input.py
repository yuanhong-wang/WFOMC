"""Input preparation for the recursive algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wfomc.arithmetic import ArithmeticValue
from wfomc.algo.core import AlgoInput, AlgoOptions
from wfomc.cell_graph import CellGraphComponent, CellGraphData, build_cell_graphs
from wfomc.cell_graph.staging import (
    build_structural_branch,
    rebind_component,
)
from wfomc.problem import CompiledBranchInstance
from wfomc.engine.features import FeatureSet

if TYPE_CHECKING:
    from wfomc.engine.compilation import CompiledReducedProblem
    from wfomc.fol import Formula


@dataclass(frozen=True)
class RecursiveInput(AlgoInput):
    components: tuple[CellGraphComponent, ...] = ()
    domain_size: int = 0


@dataclass(frozen=True)
class RecursiveInputTemplate:
    """Reusable ordered cell graphs for recursive solving."""

    components: tuple[CellGraphComponent, ...]


def build_input(
    reduced: CompiledBranchInstance,
    *,
    options: AlgoOptions,
    features: FeatureSet,
    cell_formulas: tuple["Formula", ...] | None = None,
) -> RecursiveInput:
    leq = features.leq_predicate
    components = tuple(
        _component(data, graph_weight)
        for data, graph_weight in build_cell_graphs(
            reduced.sentence,
            reduced.weights,
            reduced.arithmetic,
            leq_pred=leq,
            cell_formulas=cell_formulas,
        )
    )
    return RecursiveInput(
        algo=None,
        options=options,
        arithmetic=reduced.arithmetic,
        components=components,
        domain_size=len(reduced.domain),
    )


def build_input_template(
    compiled: "CompiledReducedProblem",
    *,
    input_variant: object,
    options: AlgoOptions,
) -> RecursiveInputTemplate:
    """Build recursive cell graphs without binding a domain."""

    structural, cell_formulas = build_structural_branch(compiled, input_variant)
    built = build_input(
        structural,
        options=options,
        features=compiled.feature_set,
        cell_formulas=cell_formulas,
    )
    return RecursiveInputTemplate(built.components)


def instantiate_input_template(
    template: RecursiveInputTemplate,
    concrete: CompiledBranchInstance,
    *,
    options: AlgoOptions,
) -> RecursiveInput:
    """Bind recursive graph weights for one concrete domain."""

    return RecursiveInput(
        algo=None,
        options=options,
        arithmetic=concrete.arithmetic,
        components=tuple(
            rebind_component(component, concrete, include_evidence=False)
            for component in template.components
        ),
        domain_size=len(concrete.domain),
    )


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
    "build_input",
    "build_input_template",
    "instantiate_input_template",
]
