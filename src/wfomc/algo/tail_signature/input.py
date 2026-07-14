"""Input contract owned by the tail-signature algorithm."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from wfomc.algo.core import AlgoInput, AlgoName, AlgoOptions
from wfomc.arithmetic import ArithmeticValue
from wfomc.cardinality_constraints import CardinalityConstraints
from wfomc.cell_graph import CellGraphData, build_cell_graphs
from wfomc.problem import CompiledProblem

if TYPE_CHECKING:
    from wfomc.evidence.profile import ProfileCapacityConstraint
    from wfomc.fol.syntax import Predicate


@dataclass(frozen=True)
class TailSignatureComponent:
    w_tables: tuple[tuple[ArithmeticValue, ...], ...] = ()
    r_matrix: tuple[tuple[ArithmeticValue, ...], ...] = ()
    graph_weight: ArithmeticValue | int = 1


@dataclass(frozen=True)
class TailSignatureProfileCapacities:
    profile_sizes: tuple[int, ...] = ()
    required_unary_predicates: frozenset[Predicate] = frozenset()
    assignment_count: Fraction = Fraction(1)


@dataclass(frozen=True)
class TailSignaturePolynomialContext:
    cardinality_constraints: CardinalityConstraints


@dataclass(frozen=True)
class TailSignatureInput(AlgoInput):
    components: tuple[TailSignatureComponent, ...] = ()
    domain_size: int = 0
    unary_profile_capacities: TailSignatureProfileCapacities | None = None
    polynomial_context: TailSignaturePolynomialContext | None = None
    engine_options: Mapping[str, bool] | None = None


def build_input(
    reduced: CompiledProblem,
    *,
    options: AlgoOptions,
    cardinality_constraints: CardinalityConstraints,
) -> TailSignatureInput:
    components = tuple(
        _component(data, graph_weight, len(reduced.domain))
        for data, graph_weight in build_cell_graphs(
            reduced.sentence,
            reduced.weights,
            reduced.arithmetic,
        )
    )
    return TailSignatureInput(
        algo=AlgoName.TAIL_SIGNATURE,
        options=options,
        arithmetic=reduced.arithmetic,
        components=components,
        domain_size=len(reduced.domain),
        unary_profile_capacities=_profile_capacities(
            reduced.profile_capacity_constraint
        ),
        polynomial_context=_polynomial_context(cardinality_constraints),
        engine_options={
            "use_symmetric_cliques": True,
            "use_small_tail_shape_summary_reject": True,
        },
    )


def _profile_capacities(
    constraint: "ProfileCapacityConstraint | None",
) -> TailSignatureProfileCapacities | None:
    if constraint is None or constraint.is_empty:
        return None
    return TailSignatureProfileCapacities(
        profile_sizes=tuple(profile.size for profile in constraint.profiles),
        required_unary_predicates=frozenset(
            literal.predicate
            for profile in constraint.profiles
            for literal in profile.literals
        ),
        assignment_count=constraint.assignment_count,
    )


def _polynomial_context(
    cardinality_constraints: CardinalityConstraints,
) -> TailSignaturePolynomialContext | None:
    if cardinality_constraints.is_empty:
        return None
    return TailSignaturePolynomialContext(cardinality_constraints)


def _component(
    data: CellGraphData,
    graph_weight: ArithmeticValue,
    domain_size: int,
) -> TailSignatureComponent:
    pair_weights = data.pair_weights()
    return TailSignatureComponent(
        w_tables=_local_weight_tables(
            data.cell_weights,
            pair_weights,
            domain_size,
        ),
        r_matrix=pair_weights,
        graph_weight=graph_weight,
    )


def _local_weight_tables(
    cell_weights: tuple[ArithmeticValue, ...],
    pair_weights: tuple[tuple[ArithmeticValue, ...], ...],
    domain_size: int,
) -> tuple[tuple[ArithmeticValue, ...], ...]:
    return tuple(
        tuple(
            (cell_weight**count)
            * (pair_weights[idx][idx] ** (count * (count - 1) // 2))
            for count in range(domain_size + 1)
        )
        for idx, cell_weight in enumerate(cell_weights)
    )


__all__ = [
    "TailSignatureComponent",
    "TailSignatureInput",
    "TailSignaturePolynomialContext",
    "TailSignatureProfileCapacities",
    "build_input",
]
