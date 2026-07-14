"""Reduce cardinality constraints into marker-weighted problems."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Mapping

from flint import fmpq, fmpq_mpoly

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.cardinality_constraints import (
    CardinalityConstraints,
)
from wfomc.problem import ReducedProblem
from wfomc.reduction.core import ProblemWithDecoder
from wfomc.weights import collect_output_weight_variables, compile_weight_mapping

if TYPE_CHECKING:
    from wfomc.algo.core import AlgoOptions


def reduce_cardinality_constraints(
    problem: ReducedProblem,
    *,
    options: AlgoOptions,
) -> ReducedProblem | ProblemWithDecoder:
    """Move cardinality constraints into positive predicate weights."""

    constraints = problem.cardinality_constraints
    if constraints.is_empty:
        return problem

    predicates = sorted(
        {
            term.predicate
            for constraint in constraints.constraints
            for term in constraint.terms
        },
        key=str,
    )
    predicate_markers = tuple(
        (predicate, f"__wfomc_cardinality_{index}")
        for index, predicate in enumerate(predicates)
    )
    marker_names = tuple(marker for _predicate, marker in predicate_markers)
    user_symbols = collect_output_weight_variables(problem)
    arithmetic = ArithmeticContext(
        backend=ArithmeticBackend.FMPQ_MPOLY,
        symbolic_variables=tuple(sorted(set(user_symbols) | set(marker_names))),
        output_symbols=user_symbols,
    )
    weights = compile_weight_mapping(dict(problem.weights), arithmetic)
    for predicate, marker in predicate_markers:
        positive, negative = weights.get(
            predicate,
            (arithmetic.one(), arithmetic.one()),
        )
        weights[predicate] = (positive * arithmetic.symbol(marker), negative)

    reduced = replace(
        problem,
        weights=dict(sorted(weights.items(), key=lambda item: str(item[0]))),
        cardinality_constraints=CardinalityConstraints(),
        internal_weight_symbols=tuple(
            sorted(set(problem.internal_weight_symbols) | set(marker_names))
        ),
    )

    def decode(result: object, **kwargs: object) -> object:
        arithmetic = kwargs.get("arithmetic")
        if not isinstance(arithmetic, ArithmeticContext):
            raise TypeError("cardinality decoder requires ArithmeticContext")
        return _decode_result(
            result,
            constraints,
            predicate_markers,
            arithmetic,
        )

    return ProblemWithDecoder(reduced, decode)


def _decode_result(
    value: object,
    constraints: CardinalityConstraints,
    predicate_markers: tuple[tuple[object, str], ...],
    arithmetic: ArithmeticContext,
) -> object:
    if isinstance(value, fmpq):
        return value if _valid_degrees(constraints, {}) else arithmetic.zero()
    if not isinstance(value, fmpq_mpoly):
        raise TypeError(f"Unsupported cardinality result type: {type(value)}")

    names = value.context().names()
    marker_indices = {
        marker: names.index(marker)
        for _predicate, marker in predicate_markers
        if marker in names
    }
    valid_terms = {
        monomial: coefficient
        for monomial, coefficient in value.to_dict().items()
        if _valid_degrees(
            constraints,
            {
                predicate: monomial[marker_indices[marker]]
                for predicate, marker in predicate_markers
                if marker in marker_indices
            },
        )
    }
    filtered = value.context().from_dict(valid_terms)
    filtered = filtered.subs(
        {marker: 1 for _predicate, marker in predicate_markers}
    )
    # Keep the expanded arithmetic ring until earlier reduction decoders have
    # applied their correction factors.  The engine projects away marker
    # symbols once the complete decoder chain has finished.
    return filtered


def _valid_degrees(
    constraints: CardinalityConstraints,
    degrees: Mapping[object, int],
) -> bool:
    return all(
        constraint.accepts(
            sum(
                term.coefficient * degrees.get(term.predicate, 0)
                for term in constraint.terms
            )
        )
        for constraint in constraints.constraints
    )


__all__ = ["reduce_cardinality_constraints"]
