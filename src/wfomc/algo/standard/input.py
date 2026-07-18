"""Standard WFOMC input contract and materialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
from wfomc.cell_graph.staging import (
    build_structural_branch,
    rebind_component,
)
from wfomc.problem import CompiledBranchInstance

if TYPE_CHECKING:
    from wfomc.engine.compilation import CompiledReducedProblem
    from wfomc.fol import Formula


@dataclass(frozen=True)
class StandardInput(AlgoInput):
    components: tuple[CellGraphComponent, ...]
    domain_size: int


@dataclass(frozen=True)
class StandardInputTemplate:
    """Reusable scalar cell graphs for the Standard algorithm."""

    components: tuple[CellGraphComponent, ...]


def build_input(
    reduced: CompiledBranchInstance,
    *,
    options: AlgoOptions,
    cell_formulas: tuple["Formula", ...] | None = None,
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
            cell_formulas=(
                cell_formulas
                if cell_formulas is not None
                else profile_cell_formulas(reduced.profile_capacity_constraint)
            ),
        )
    )
    return StandardInput(
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
) -> StandardInputTemplate:
    """Build Standard cell graphs without binding a concrete domain."""

    structural, cell_formulas = build_structural_branch(compiled, input_variant)
    built = build_input(
        structural,
        options=options,
        cell_formulas=cell_formulas,
    )
    return StandardInputTemplate(built.components)


def instantiate_input_template(
    template: StandardInputTemplate,
    concrete: CompiledBranchInstance,
    *,
    options: AlgoOptions,
) -> StandardInput:
    """Bind arithmetic and unary-evidence capacities for one domain."""

    return StandardInput(
        algo=None,
        options=options,
        arithmetic=concrete.arithmetic,
        components=tuple(
            rebind_component(component, concrete, include_evidence=True)
            for component in template.components
        ),
        domain_size=len(concrete.domain),
    )


def _standard_component_from_cell_graph(
    reduced: CompiledBranchInstance,
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
    "StandardInputTemplate",
    "build_input",
    "build_input_template",
    "instantiate_input_template",
]
