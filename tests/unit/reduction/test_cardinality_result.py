from __future__ import annotations

import pytest
from flint import fmpq_poly

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.cardinality_constraints import (
    CardinalityConstraints,
    CardinalityTerm,
    Comparator,
    LinearCardinalityConstraint,
)
from wfomc.reduction.cardinality import decode_cardinality_result
from wfomc.stages import CardinalityMarkerEncoding


PREDICATE = "P"
OTHER_PREDICATE = "Q"
MARKER = "z"
ARITHMETIC = ArithmeticContext(ArithmeticBackend.FMPQ_POLY, (MARKER,))


def _constraint(
    *terms: CardinalityTerm,
    comparator: Comparator,
    rhs: int,
) -> LinearCardinalityConstraint:
    return LinearCardinalityConstraint(terms, comparator, rhs)


def test_univariate_predicate_marker_supports_multiple_constraints() -> None:
    constraints = CardinalityConstraints(
        (
            _constraint(
                CardinalityTerm(PREDICATE),
                comparator=Comparator.GE,
                rhs=1,
            ),
            _constraint(
                CardinalityTerm(PREDICATE),
                comparator=Comparator.LE,
                rhs=2,
            ),
        )
    )

    actual = decode_cardinality_result(
        fmpq_poly([1, 3, 3, 1]),
        constraints,
        ((PREDICATE, MARKER, 1),),
        ARITHMETIC,
        marker_encoding=CardinalityMarkerEncoding.PER_PREDICATE,
    )

    assert actual == fmpq_poly([6])


def test_univariate_predicate_marker_respects_negative_coefficients() -> None:
    constraints = CardinalityConstraints(
        (
            _constraint(
                CardinalityTerm(PREDICATE, -1),
                comparator=Comparator.LE,
                rhs=-1,
            ),
        )
    )

    actual = decode_cardinality_result(
        fmpq_poly([1, 3, 3, 1]),
        constraints,
        ((PREDICATE, MARKER, 1),),
        ARITHMETIC,
        marker_encoding=CardinalityMarkerEncoding.PER_PREDICATE,
    )

    assert actual == fmpq_poly([7])


def test_shared_linear_form_marker_checks_its_single_constraint() -> None:
    constraints = CardinalityConstraints(
        (
            _constraint(
                CardinalityTerm(PREDICATE),
                CardinalityTerm(OTHER_PREDICATE),
                comparator=Comparator.LE,
                rhs=1,
            ),
        )
    )

    actual = decode_cardinality_result(
        fmpq_poly([1, 2, 1]),
        constraints,
        (
            (PREDICATE, MARKER, 1),
            (OTHER_PREDICATE, MARKER, 1),
        ),
        ARITHMETIC,
        marker_encoding=CardinalityMarkerEncoding.SHARED_LINEAR_FORM,
    )

    assert actual == fmpq_poly([3])


def test_shared_linear_form_marker_rejects_multiple_constraints() -> None:
    constraints = CardinalityConstraints(
        (
            _constraint(
                CardinalityTerm(PREDICATE),
                CardinalityTerm(OTHER_PREDICATE),
                comparator=Comparator.LE,
                rhs=1,
            ),
            _constraint(
                CardinalityTerm(PREDICATE),
                comparator=Comparator.GE,
                rhs=1,
            ),
        )
    )

    with pytest.raises(TypeError, match="one marker and one constraint"):
        decode_cardinality_result(
            fmpq_poly([1, 2, 1]),
            constraints,
            (
                (PREDICATE, MARKER, 1),
                (OTHER_PREDICATE, MARKER, 1),
            ),
            ARITHMETIC,
            marker_encoding=CardinalityMarkerEncoding.SHARED_LINEAR_FORM,
        )
