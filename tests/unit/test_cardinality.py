from __future__ import annotations

import pytest

from wfomc.cardinality_constraints import (
    CardinalityConstraints,
    CardinalityTerm,
    Comparator,
    LinearCardinalityConstraint,
)


def test_linear_cardinality_constraint_uses_exact_integer_terms():
    constraint = LinearCardinalityConstraint(
        terms=(
            CardinalityTerm("P", 2),
            CardinalityTerm("Q", -1),
        ),
        comparator=Comparator.LE,
        rhs=3,
    )

    assert constraint.terms[0].coefficient == 2
    assert isinstance(constraint.terms[0].coefficient, int)
    assert constraint.terms[1].coefficient == -1
    assert constraint.comparator == Comparator.LE
    assert constraint.rhs == 3


def test_cardinality_constraints_reports_empty_state():
    assert CardinalityConstraints().is_empty
    assert not CardinalityConstraints(
        (
            LinearCardinalityConstraint(
                terms=(CardinalityTerm("P"),),
                comparator=Comparator.EQ,
                rhs=1,
            ),
        )
    ).is_empty


def test_mod_constraint_requires_a_positive_modulus():
    with pytest.raises(ValueError, match="positive modulus"):
        LinearCardinalityConstraint(
            terms=(CardinalityTerm("P"),),
            comparator=Comparator.MOD,
            rhs=0,
        )


def test_non_mod_constraint_rejects_modulus():
    with pytest.raises(ValueError, match="Only MOD"):
        LinearCardinalityConstraint(
            terms=(CardinalityTerm("P"),),
            comparator=Comparator.EQ,
            rhs=0,
            modulus=2,
        )


def test_cache_key_includes_modulus():
    def constraints(modulus: int) -> CardinalityConstraints:
        return CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    terms=(CardinalityTerm("P"),),
                    comparator=Comparator.MOD,
                    rhs=0,
                    modulus=modulus,
                ),
            )
        )

    assert constraints(2).cache_key_parts() != constraints(3).cache_key_parts()


def test_constraint_accepts_order_comparisons():
    cases = (
        (Comparator.EQ, 3, 3, True),
        (Comparator.NE, 3, 3, False),
        (Comparator.LT, 3, 2, True),
        (Comparator.LE, 3, 3, True),
        (Comparator.GT, 3, 4, True),
        (Comparator.GE, 3, 3, True),
    )

    for comparator, rhs, value, expected in cases:
        constraint = LinearCardinalityConstraint((), comparator, rhs)
        assert constraint.accepts(value) is expected, comparator


def test_constraint_accepts_normalized_mod_comparison():
    constraint = LinearCardinalityConstraint((), Comparator.MOD, 4, modulus=3)
    assert constraint.accepts(1)
    assert not constraint.accepts(2)


def test_multinomial_coefficient_supports_the_empty_domain():
    from wfomc.multinomial import multinomial_coefficient

    assert multinomial_coefficient(()) == 1
    assert multinomial_coefficient((0,)) == 1
