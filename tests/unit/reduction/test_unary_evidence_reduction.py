"""Tests for unary-evidence reductions (profile-capacity and cardinality paths)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext

from wfomc.cardinality_constraints import Comparator
from wfomc.evidence import (
    Evidence,
    GroundUnaryLiteral,
    UnaryEvidence,
)
from wfomc.evidence.profile import ProfileCapacityConstraint
from wfomc.fol import true as _true
from wfomc.problem import Problem, ReducedProblem
from wfomc.reduction import (
    begin_reduction,
    build_unary_ccs_encoding,
    reduce_unary_evidence_to_cardinality_constraints,
    reduce_unary_evidence_to_profile_capacity,
)


def _arithmetic() -> ArithmeticContext:
    return ArithmeticContext(ArithmeticBackend.FMPQ)


def _problem_with_evidence() -> ReducedProblem:
    return begin_reduction(
        Problem(
            sentence=_true(),
            domain=frozenset({"a", "b", "c"}),
            evidence=Evidence(
                unary=UnaryEvidence(
                    (
                        GroundUnaryLiteral("P", "a", True),
                        GroundUnaryLiteral("Q", "a", False),
                        GroundUnaryLiteral("P", "b", True),
                    )
                ),
            ),
        )
    )


def _empty_evidence_problem() -> ReducedProblem:
    return begin_reduction(Problem(sentence=_true(), domain=frozenset({"a", "b"})))


# --- profile-capacity path ---


def test_profile_capacity_empty_evidence_returns_same_problem():
    problem = _empty_evidence_problem()
    assert reduce_unary_evidence_to_profile_capacity(problem) is problem


def test_profile_capacity_nonempty_evidence_produces_one_constraint():
    reduced = reduce_unary_evidence_to_profile_capacity(_problem_with_evidence())
    constraint = reduced.profile_capacity_constraint
    assert isinstance(constraint, ProfileCapacityConstraint)
    # three distinct profiles: empty, {P}, {P, ~Q}
    assert len(constraint.profiles) == 3


def test_profile_capacity_clears_unary_evidence():
    reduced = reduce_unary_evidence_to_profile_capacity(_problem_with_evidence())
    assert reduced.evidence.unary.is_empty


def test_profile_capacity_existing_constraint_with_evidence_raises():

    problem = replace(
        begin_reduction(
            Problem(
                sentence=_true(),
                domain=frozenset({"a", "b"}),
                evidence=Evidence(
                    unary=UnaryEvidence((GroundUnaryLiteral("P", "a", True),)),
                ),
            )
        ),
        profile_capacity_constraint=ProfileCapacityConstraint(
            profiles=(), domain_size=0
        ),
    )
    with pytest.raises(ValueError, match="profile_capacity_constraint"):
        reduce_unary_evidence_to_profile_capacity(problem)


def test_profile_capacity_existing_constraint_with_empty_evidence_is_noop():
    problem = replace(
        begin_reduction(Problem(sentence=_true(), domain=frozenset({"a"}))),
        profile_capacity_constraint=ProfileCapacityConstraint(
            profiles=(), domain_size=0
        ),
    )
    # empty evidence short-circuits before the conflict check
    assert reduce_unary_evidence_to_profile_capacity(problem) is problem


# --- cardinality-constraint path ---


def test_cardinality_constraints_empty_evidence_returns_same_problem():
    problem = _empty_evidence_problem()
    assert reduce_unary_evidence_to_cardinality_constraints(problem) is problem


def test_cardinality_constraints_clears_unary_evidence():
    reduced = reduce_unary_evidence_to_cardinality_constraints(
        _problem_with_evidence()
    ).problem
    assert reduced.evidence.unary.is_empty


def test_cardinality_constraints_returns_decoder_and_patches_problem():
    problem = _problem_with_evidence()
    branch = reduce_unary_evidence_to_cardinality_constraints(problem)
    reduced = branch.problem
    assert branch.decoder(12, arithmetic=_arithmetic()) == 2
    # one EQ cardinality constraint per evidence profile
    assert len(reduced.cardinality_constraints.constraints) == 3
    assert all(
        c.comparator == Comparator.EQ
        for c in reduced.cardinality_constraints.constraints
    )


def test_cardinality_constraints_preserves_existing_plan_constraints():
    from wfomc.cardinality_constraints import (
        CardinalityConstraints,
        LinearCardinalityConstraint,
        CardinalityTerm,
    )

    existing = CardinalityConstraints(
        (
            LinearCardinalityConstraint(
                terms=(CardinalityTerm("R", 1),), comparator=Comparator.LE, rhs=2
            ),
        )
    )
    problem = begin_reduction(
        Problem(
            sentence=_true(),
            domain=frozenset({"a", "b"}),
            evidence=Evidence(
                unary=UnaryEvidence((GroundUnaryLiteral("P", "a", True),)),
            ),
            cardinality_constraints=existing,
        )
    )
    reduced = reduce_unary_evidence_to_cardinality_constraints(problem).problem
    # one pre-existing + one per evidence profile. P(a) over {a, b} yields two
    # profiles ({P} for a, empty for b), so 1 + 2 == 3 constraints.
    assert len(reduced.cardinality_constraints.constraints) == 3


# --- build_unary_ccs_encoding helper ---


def test_build_unary_ccs_encoding():
    evidence = UnaryEvidence(
        (
            GroundUnaryLiteral("P", "a", True),
            GroundUnaryLiteral("P", "b", True),
            GroundUnaryLiteral("Q", "c", False),
        )
    )
    encoding = build_unary_ccs_encoding(evidence, frozenset({"a", "b", "c"}))
    assert encoding.formula_patch is not None
    assert len(encoding.cardinality_constraints) == 2
    assert encoding.correction_factor == 3
