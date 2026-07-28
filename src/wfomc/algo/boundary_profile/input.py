"""Input contract and decomposition planning for Boundary-Profile DP."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from wfomc.algo.core import AlgoInput, ReducedInputTemplate
from wfomc.arithmetic import ArithmeticValue
from wfomc.cell_graph import (
    build_cell_graphs,
    required_profile_predicates,
)
from wfomc.cell_graph.staging import select_cell_graph_structure
from wfomc.options import BoundaryProfileOptions
from wfomc.stages import CompiledBranchInstance, CompiledReducedBranch

from .plan import (
    BoundaryProfilePlan,
    BoundaryProfileTree,
    build_boundary_profile_tree,
    materialize_boundary_profile_plan,
)


@dataclass(frozen=True)
class BoundaryProfileComponent:
    """One master-sum branch paired with its selected decomposition plan."""

    cell_weights: tuple[ArithmeticValue, ...] = ()
    w_tables: tuple[tuple[ArithmeticValue, ...], ...] = ()
    r_matrix: tuple[tuple[ArithmeticValue, ...], ...] = ()
    graph_weight: ArithmeticValue | int = 1
    plan: BoundaryProfilePlan | None = None


@dataclass(frozen=True)
class BoundaryProfileInput(AlgoInput):
    """Fully materialized input for the native Boundary-Profile solver."""

    components: tuple[BoundaryProfileComponent, ...] = ()
    domain_size: int = 0


@dataclass(frozen=True)
class BoundaryProfileInputTemplate(ReducedInputTemplate):
    """Reusable scalar cell graphs for Boundary-Profile planning."""

    components: tuple["BoundaryProfileTemplateComponent", ...]

    def instantiate(
        self,
        concrete: CompiledBranchInstance,
    ) -> BoundaryProfileInput:
        """Materialize local tables and choose plans for one domain."""

        domain_size = len(concrete.domain)
        components = []
        for template in self.components:
            components.append(template.instantiate(concrete))
        return BoundaryProfileInput(
            arithmetic=concrete.arithmetic,
            components=tuple(components),
            domain_size=domain_size,
        )


@dataclass(frozen=True)
class BoundaryProfileTemplateComponent:
    """Domain-free scalar cell graph and its cached BP tree."""

    cell_weights: tuple[ArithmeticValue, ...]
    r_matrix: tuple[tuple[ArithmeticValue, ...], ...]
    graph_weight: ArithmeticValue | int
    tree: BoundaryProfileTree

    def instantiate(
        self,
        concrete: CompiledBranchInstance,
    ) -> BoundaryProfileComponent:
        """Bind numeric values and local tables without rebuilding the tree."""

        arithmetic = concrete.arithmetic
        domain_size = len(concrete.domain)
        cell_weights = tuple(arithmetic.coerce(value) for value in self.cell_weights)
        r_matrix = tuple(
            tuple(arithmetic.coerce(value) for value in row)
            for row in self.r_matrix
        )
        component = BoundaryProfileComponent(
            cell_weights=cell_weights,
            w_tables=local_weight_tables(
                cell_weights,
                r_matrix,
                domain_size,
                arithmetic,
            ),
            r_matrix=r_matrix,
            graph_weight=arithmetic.coerce(self.graph_weight),
        )
        return BoundaryProfileComponent(
            cell_weights=component.cell_weights,
            w_tables=component.w_tables,
            r_matrix=component.r_matrix,
            graph_weight=component.graph_weight,
            plan=materialize_boundary_profile_plan(
                self.tree,
                component,
                domain_size,
                arithmetic,
            ),
        )


def build_input_template(
    compiled: CompiledReducedBranch,
    *,
    input_variant: Hashable,
    options: BoundaryProfileOptions,
) -> BoundaryProfileInputTemplate:
    """Build domain-free cell weights and one reusable tree per component."""

    profile, cell_formulas = select_cell_graph_structure(compiled, input_variant)
    components = []
    for data, graph_weight in build_cell_graphs(
        compiled.sentence,
        compiled.weights,
        compiled.arithmetic,
        required_unary_preds=required_profile_predicates(profile),
        cell_formulas=cell_formulas,
    ):
        r_matrix = data.pair_weights()
        components.append(
            BoundaryProfileTemplateComponent(
                cell_weights=data.cell_weights,
                r_matrix=r_matrix,
                graph_weight=graph_weight,
                tree=build_boundary_profile_tree(
                    data.cell_weights,
                    r_matrix,
                    compiled.arithmetic,
                    reference_domain_size=options.tree_reference_domain_size,
                    planner_strategy=options.planner_strategy,
                ),
            )
        )

    return BoundaryProfileInputTemplate(tuple(components))


def local_weight_tables(
    cell_weights: tuple[ArithmeticValue, ...],
    pair_weights: tuple[tuple[ArithmeticValue, ...], ...],
    domain_size: int,
    arithmetic,
) -> tuple[tuple[ArithmeticValue, ...], ...]:
    """Return ``W_i(k) = w_i^k R_ii^choose(k, 2)`` for every cell."""

    return tuple(
        tuple(
            arithmetic.multiply(
                arithmetic.power(cell_weight, count),
                arithmetic.power(
                    pair_weights[index][index],
                    count * (count - 1) // 2,
                ),
            )
            for count in range(domain_size + 1)
        )
        for index, cell_weight in enumerate(cell_weights)
    )


__all__ = [
    "BoundaryProfileComponent",
    "BoundaryProfileInput",
    "BoundaryProfileInputTemplate",
    "BoundaryProfileTemplateComponent",
    "build_input_template",
    "local_weight_tables",
]
