"""Decode marker-weighted cardinality results."""

from __future__ import annotations

from typing import Mapping

from flint import fmpq, fmpq_mpoly, fmpq_poly

from wfomc.arithmetic import ArithmeticContext
from wfomc.cardinality_constraints import (
    CardinalityConstraints,
)

def decode_cardinality_result(
    value: object,
    constraints: CardinalityConstraints,
    predicate_markers: tuple[tuple[object, str], ...],
    arithmetic: ArithmeticContext,
) -> object:
    if isinstance(value, fmpq):
        return value if _valid_degrees(constraints, {}) else arithmetic.zero()
    if isinstance(value, fmpq_poly):
        if len(predicate_markers) != 1:
            raise TypeError(
                "univariate cardinality result requires exactly one marker"
            )
        predicate, _marker = predicate_markers[0]
        accepted = fmpq(0)
        for degree, coefficient in enumerate(value.coeffs()):
            if _valid_degrees(constraints, {predicate: degree}):
                accepted += coefficient
        # Keep the selected backend until all earlier reduction decoders have
        # applied their factors; the engine projects this constant afterwards.
        return fmpq_poly([accepted])
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


__all__ = ["decode_cardinality_result"]
