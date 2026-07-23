"""Cardinality marker encoding and result decoding."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Mapping

from wfomc.cardinality_constraints import (
    CardinalityConstraints,
    Comparator,
    LinearCardinalityConstraint,
)
from wfomc.stages import (
    CardinalityDecoderSpec,
    DomainExpr,
    ReducedCardinalityConstraint,
    ReducedProblem,
)
from wfomc.weights import collect_output_weight_variables

if TYPE_CHECKING:
    from wfomc.arithmetic import ArithmeticContext


def encode_cardinality_constraints(problem: ReducedProblem) -> ReducedProblem:
    """Prepare generating-function markers and a matching result decoder."""

    constraints = problem.cardinality_constraints
    if not constraints:
        return problem
    user_symbols = collect_output_weight_variables(
        problem.weights,
        problem.internal_weight_symbols,
    )
    used_symbols = set(user_symbols) | set(problem.internal_weight_symbols)

    def fresh_marker(index: int) -> str:
        marker = f"__wfomc_cardinality_{index}"
        while marker in used_symbols:
            marker += "_"
        used_symbols.add(marker)
        return marker

    predicate_markers: list[tuple[object, str, int]] = []
    limits = dict(problem.internal_weight_degree_limits)
    shared_linear_form_encoding = False
    coefficients: dict[object, int] = {}
    if len(constraints) == 1:
        for term in constraints[0].terms:
            coefficients[term.predicate] = (
                coefficients.get(term.predicate, 0) + term.coefficient
            )
        shared_linear_form_encoding = (
            any(coefficient > 0 for coefficient in coefficients.values())
            and all(coefficient >= 0 for coefficient in coefficients.values())
        )
    if shared_linear_form_encoding:
        # A single linear form needs only one generating-function variable:
        # multiplying predicate P's positive weight by z**a makes z's degree
        # equal sum(a * |P|).  This is crucial for aggregate constraints such
        # as |E1| + |E2| + |E3| = 3n.
        constraint = constraints[0]
        marker = fresh_marker(0)
        predicate_markers.extend(
            (predicate, marker, coefficient)
            for predicate, coefficient in sorted(
                coefficients.items(), key=lambda item: str(item[0])
            )
            if coefficient > 0
        )
        if (
            len(predicate_markers) == len(coefficients)
            and constraint.comparator in (Comparator.EQ, Comparator.LE, Comparator.LT)
        ):
            bound = (
                constraint.rhs - 1
                if constraint.comparator is Comparator.LT
                else constraint.rhs
            )
            limits[marker] = (
                bound
                if marker not in limits
                else DomainExpr.minimum(limits[marker], bound)
            )
    else:
        predicates = sorted(
            {
                term.predicate
                for constraint in constraints
                for term in constraint.terms
            },
            key=str,
        )
        for index, predicate in enumerate(predicates):
            predicate_markers.append((predicate, fresh_marker(index), 1))

    predicate_markers_tuple = tuple(predicate_markers)
    marker_names = tuple(
        sorted({marker for _predicate, marker, _exponent in predicate_markers})
    )
    if not shared_linear_form_encoding:
        degree_bounds = _safe_predicate_upper_bounds(constraints)
        for predicate, marker, _exponent in predicate_markers_tuple:
            bound = degree_bounds.get(predicate)
            if bound is None:
                continue
            limits[marker] = (
                bound
                if marker not in limits
                else DomainExpr.minimum(limits[marker], bound)
            )
    return replace(
        problem,
        cardinality_constraints=(),
        internal_weight_symbols=tuple(
            sorted(set(problem.internal_weight_symbols) | set(marker_names))
        ),
        internal_weight_degree_limits=tuple(sorted(limits.items())),
        decoder_spec=problem.decoder_spec.append(
            CardinalityDecoderSpec(constraints, predicate_markers_tuple)
        ),
    )


def to_reduced_cardinality_constraint(
    constraint: LinearCardinalityConstraint,
) -> ReducedCardinalityConstraint:
    """Lift a source constraint into the domain-parametric reduction form."""

    return ReducedCardinalityConstraint(
        terms=constraint.terms,
        comparator=constraint.comparator,
        rhs=DomainExpr.constant(constraint.rhs),
        modulus=constraint.modulus,
    )


def decode_cardinality_result(
    value: object,
    constraints: CardinalityConstraints,
    predicate_markers: tuple[tuple[object, str, int], ...],
    arithmetic: ArithmeticContext,
) -> object:
    from flint import fmpq, fmpq_mpoly, fmpq_poly, fmpq_series

    if isinstance(value, fmpq):
        return value if _valid_degrees(constraints, {}) else arithmetic.zero()
    if isinstance(value, (fmpq_poly, fmpq_series)):
        markers = {marker for _predicate, marker, _exponent in predicate_markers}
        if len(markers) != 1 or len(constraints.constraints) != 1:
            raise TypeError(
                "univariate cardinality result requires exactly one linear form"
            )
        constraint = constraints.constraints[0]
        accepted = fmpq(0)
        for degree, coefficient in enumerate(value.coeffs()):
            if constraint.accepts(degree):
                accepted += coefficient
        # Keep the selected backend until all earlier reduction decoders have
        # applied their factors; the engine projects this constant afterwards.
        if isinstance(value, fmpq_series):
            return fmpq_series([accepted], prec=value.prec)
        return fmpq_poly([accepted])
    if not isinstance(value, fmpq_mpoly):
        raise TypeError(f"Unsupported cardinality result type: {type(value)}")

    names = value.context().names()
    marker_indices = {
        marker: names.index(marker)
        for _predicate, marker, _exponent in predicate_markers
        if marker in names
    }
    shared_linear_form = (
        len(marker_indices) == 1
        and len(constraints.constraints) == 1
        and len(predicate_markers) > 1
    )
    valid_terms = {
        monomial: coefficient
        for monomial, coefficient in value.to_dict().items()
        if (
            constraints.constraints[0].accepts(
                monomial[next(iter(marker_indices.values()))]
            )
            if shared_linear_form
            else _valid_degrees(
                constraints,
                {
                    predicate: monomial[marker_indices[marker]] // exponent
                    for predicate, marker, exponent in predicate_markers
                    if marker in marker_indices
                },
            )
        )
    }
    filtered = value.context().from_dict(valid_terms)
    filtered = filtered.subs(
        {
            marker: 1
            for _predicate, marker, _exponent in predicate_markers
        }
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


def _safe_predicate_upper_bounds(
    constraints: tuple[ReducedCardinalityConstraint, ...],
) -> dict[object, DomainExpr]:
    bounds: dict[object, list[DomainExpr]] = {}
    for constraint in constraints:
        if constraint.comparator in (Comparator.EQ, Comparator.LE):
            total = constraint.rhs
        elif constraint.comparator is Comparator.LT:
            total = constraint.rhs - 1
        else:
            continue
        coefficients: dict[object, int] = {}
        for term in constraint.terms:
            coefficients[term.predicate] = (
                coefficients.get(term.predicate, 0) + term.coefficient
            )
        if any(coefficient < 0 for coefficient in coefficients.values()):
            continue
        for predicate, coefficient in coefficients.items():
            if coefficient > 0:
                bounds.setdefault(predicate, []).append(total // coefficient)
    return {
        predicate: DomainExpr.minimum(*predicate_bounds)
        for predicate, predicate_bounds in bounds.items()
    }


__all__ = [
    "decode_cardinality_result",
    "encode_cardinality_constraints",
    "to_reduced_cardinality_constraint",
]
