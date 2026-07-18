"""Input contract owned by the incremental algorithm."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import TYPE_CHECKING

from wfomc.arithmetic import ArithmeticValue
from wfomc.algo.core import AlgoInput, AlgoOptions
from wfomc.cell_graph import (
    CellGraphComponent,
    CellGraphData,
    PairWeightMatrix,
    build_cell_graphs,
    materialize_cell_evidence,
    profile_cell_formulas,
    required_profile_predicates,
)
from wfomc.cell_graph.staging import (
    build_structural_branch,
    rebind_component,
    rebind_matrix,
)
from wfomc.fol import Predicate
from wfomc.problem import CompiledBranchInstance
from wfomc.engine.features import FeatureSet

if TYPE_CHECKING:
    from wfomc.engine.compilation import CompiledReducedProblem
    from wfomc.fol import Formula


@dataclass(frozen=True)
class OrderedCellGraphComponent(CellGraphComponent):
    predk_pair_tables: Mapping[int, PairWeightMatrix] | None = None
    circular_predecessor_pair_tables: PairWeightMatrix | None = None


@dataclass(frozen=True)
class OrderedCellGraphInput(AlgoInput):
    components: tuple[OrderedCellGraphComponent, ...] = ()
    domain_size: int = 0
    predecessor_orders: tuple[int, ...] = ()
    predecessor_max_order: int = 0
    has_circular_predecessor: bool = False
    circle_len: int | None = None


@dataclass(frozen=True)
class IncrementalInputTemplate:
    """Reusable ordered cell graphs and predecessor tables."""

    components: tuple[OrderedCellGraphComponent, ...]
    predecessor_orders: tuple[int, ...]
    predecessor_max_order: int
    has_circular_predecessor: bool


def build_input(
    reduced: CompiledBranchInstance,
    *,
    options: AlgoOptions,
    features: FeatureSet,
    cell_formulas: tuple["Formula", ...] | None = None,
) -> OrderedCellGraphInput:
    leq = features.leq_predicate
    predecessor_predicates = dict(features.predecessor_predicates) or None
    circular = features.circular_predecessor_predicate
    components = tuple(
        _ordered_component(
            reduced, data, graph_weight, predecessor_predicates, circular
        )
        for data, graph_weight in build_cell_graphs(
            reduced.sentence,
            reduced.weights,
            reduced.arithmetic,
            leq_pred=leq,
            predecessor_preds=predecessor_predicates,
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
    orders = tuple(sorted((predecessor_predicates or {}).keys()))
    return OrderedCellGraphInput(
        algo=None,
        options=options,
        arithmetic=reduced.arithmetic,
        components=components,
        domain_size=len(reduced.domain),
        predecessor_orders=orders,
        predecessor_max_order=max(orders) if orders else 0,
        has_circular_predecessor=circular is not None,
        circle_len=(
            reduced.circular_order_size
            if reduced.circular_order_size is not None
            else len(reduced.domain)
        ),
    )


def build_input_template(
    compiled: "CompiledReducedProblem",
    *,
    input_variant: object,
    options: AlgoOptions,
) -> IncrementalInputTemplate:
    """Build ordered cell graphs without binding n or the circle length."""

    structural, cell_formulas = build_structural_branch(compiled, input_variant)
    built = build_input(
        structural,
        options=options,
        features=compiled.feature_set,
        cell_formulas=cell_formulas,
    )
    return IncrementalInputTemplate(
        components=built.components,
        predecessor_orders=built.predecessor_orders,
        predecessor_max_order=built.predecessor_max_order,
        has_circular_predecessor=built.has_circular_predecessor,
    )


def instantiate_input_template(
    template: IncrementalInputTemplate,
    concrete: CompiledBranchInstance,
    *,
    options: AlgoOptions,
) -> OrderedCellGraphInput:
    """Bind numeric values, profile capacities, and domain order sizes."""

    components = []
    for component in template.components:
        rebound = rebind_component(component, concrete, include_evidence=True)
        components.append(
            replace(
                rebound,
                predk_pair_tables=(
                    None
                    if component.predk_pair_tables is None
                    else {
                        order: rebind_matrix(matrix, concrete)
                        for order, matrix in component.predk_pair_tables.items()
                    }
                ),
                circular_predecessor_pair_tables=(
                    None
                    if component.circular_predecessor_pair_tables is None
                    else rebind_matrix(
                        component.circular_predecessor_pair_tables,
                        concrete,
                    )
                ),
            )
        )
    return OrderedCellGraphInput(
        algo=None,
        options=options,
        arithmetic=concrete.arithmetic,
        components=tuple(components),
        domain_size=len(concrete.domain),
        predecessor_orders=template.predecessor_orders,
        predecessor_max_order=template.predecessor_max_order,
        has_circular_predecessor=template.has_circular_predecessor,
        circle_len=(
            concrete.circular_order_size
            if concrete.circular_order_size is not None
            else len(concrete.domain)
        ),
    )


def _ordered_component(
    reduced: CompiledBranchInstance,
    data: CellGraphData,
    graph_weight: ArithmeticValue,
    predecessor_predicates: Mapping[int, Predicate] | None,
    circular: Predicate | None,
) -> OrderedCellGraphComponent:
    base_pairs = data.pair_weights()
    predecessor_tables: dict[int, PairWeightMatrix] = {}
    circular_table = None
    raw_tables = dict(data.predecessor_pair_factors)
    for order, predicate in sorted((predecessor_predicates or {}).items()):
        table = tuple(
            tuple(item.total_weight for item in row) for row in raw_tables[order]
        )
        if circular is not None and _same_predicate(predicate, circular):
            circular_table = table
        else:
            predecessor_tables[order] = table
    return OrderedCellGraphComponent(
        cells=data.cells,
        cell_weights=data.cell_weights,
        pair_weights=base_pairs,
        graph_weight=graph_weight,
        cell_evidence_allocation=materialize_cell_evidence(
            reduced.profile_capacity_constraint,
            data.cells,
        ),
        predk_pair_tables=predecessor_tables or None,
        circular_predecessor_pair_tables=circular_table,
    )


def _same_predicate(left: Predicate, right: Predicate) -> bool:
    return (left.name, left.arity) == (right.name, right.arity)


__all__ = [
    "IncrementalInputTemplate",
    "OrderedCellGraphComponent",
    "OrderedCellGraphInput",
    "build_input",
    "build_input_template",
    "instantiate_input_template",
]
