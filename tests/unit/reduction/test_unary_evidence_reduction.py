"""Tests for staged unary-evidence reductions."""

from __future__ import annotations

import pytest

from wfomc.algo import AlgoOptions, EvidenceStrategy
from wfomc.cardinality_constraints import (
    CardinalityConstraints,
    CardinalityTerm,
    Comparator,
    LinearCardinalityConstraint,
)
from wfomc.evidence import Evidence, GroundUnaryLiteral, UnaryEvidence
from wfomc.fol import true
from wfomc.problem import Problem
from wfomc.reduction import (
    CardinalityDecoderSpec,
    DivideDecoderSpec,
    ReducedProfileConstraint,
    reduce_problem,
)


def _problem_with_evidence(
    *,
    cardinality_constraints: CardinalityConstraints | None = None,
) -> Problem:
    return Problem(
        sentence=true(),
        evidence=Evidence(
            unary=UnaryEvidence(
                (
                    GroundUnaryLiteral("P", "a", True),
                    GroundUnaryLiteral("Q", "a", False),
                    GroundUnaryLiteral("P", "b", True),
                )
            ),
        ),
        cardinality_constraints=(
            cardinality_constraints
            if cardinality_constraints is not None
            else CardinalityConstraints()
        ),
    )


def test_empty_evidence_does_not_create_a_profile_constraint():
    reduced = reduce_problem(
        Problem(sentence=true()),
        AlgoOptions(evidence_strategy=EvidenceStrategy.LIFTED_PROFILES),
    )[0]

    assert reduced.profile_constraint is None
    assert reduced.evidence.unary.is_empty


def test_lifted_profiles_are_domain_free_until_instantiation():
    reduced = reduce_problem(
        _problem_with_evidence(),
        AlgoOptions(evidence_strategy=EvidenceStrategy.LIFTED_PROFILES),
    )[0]

    assert isinstance(reduced.profile_constraint, ReducedProfileConstraint)
    assert reduced.profile_constraint.profile_sizes(3) == (1, 1, 1)
    concrete = reduced.profile_constraint.instantiate(3)
    assert len(concrete.profiles) == 3
    assert concrete.assignment_count == 6


def test_lifted_profile_reduction_clears_unary_evidence():
    reduced = reduce_problem(
        _problem_with_evidence(),
        AlgoOptions(evidence_strategy=EvidenceStrategy.LIFTED_PROFILES),
    )[0]

    assert reduced.evidence.unary.is_empty


def test_lifted_profiles_group_equal_evidence_deterministically():
    problem = Problem(
        sentence=true(),
        evidence=Evidence(
            unary=UnaryEvidence(
                (
                    GroundUnaryLiteral("P", "a", True),
                    GroundUnaryLiteral("Q", "a", False),
                    GroundUnaryLiteral("P", "b", True),
                    GroundUnaryLiteral("Q", "b", False),
                    GroundUnaryLiteral("Q", "c", True),
                )
            )
        ),
    )

    profile = reduce_problem(
        problem,
        AlgoOptions(evidence_strategy=EvidenceStrategy.LIFTED_PROFILES),
    )[0].profile_constraint

    assert profile is not None
    assert profile.profile_sizes(4) == (2, 1, 1)
    assert tuple(
        tuple(sorted(map(str, literals))) for literals, _size in profile.profiles
    ) == (("P(X)", "~Q(X)"), ("Q(X)",), ())


def test_lifted_profiles_reject_conflicting_evidence():
    problem = Problem(
        sentence=true(),
        evidence=Evidence(
            unary=UnaryEvidence(
                (
                    GroundUnaryLiteral("P", "a", True),
                    GroundUnaryLiteral("P", "a", False),
                )
            )
        ),
    )

    with pytest.raises(ValueError, match="consistent"):
        reduce_problem(
            problem,
            AlgoOptions(evidence_strategy=EvidenceStrategy.LIFTED_PROFILES),
        )


def test_ccs_reduction_uses_data_only_decoder_steps():
    reduced = reduce_problem(
        _problem_with_evidence(),
        AlgoOptions(evidence_strategy=EvidenceStrategy.CCS),
    )[0]

    assert reduced.evidence.unary.is_empty
    assert len(reduced.ccs_profile_markers) == 2
    assert reduced.ccs_unmarked_size is not None
    assert reduced.ccs_unmarked_size.evaluate(3) == 1
    assert isinstance(reduced.decoder_spec.steps[0], DivideDecoderSpec)
    assert isinstance(reduced.decoder_spec.steps[-1], CardinalityDecoderSpec)


def test_ccs_reduction_preserves_source_cardinality_constraints():
    existing = CardinalityConstraints(
        (
            LinearCardinalityConstraint(
                terms=(CardinalityTerm("R", 1),),
                comparator=Comparator.LE,
                rhs=2,
            ),
        )
    )

    reduced = reduce_problem(
        _problem_with_evidence(cardinality_constraints=existing),
        AlgoOptions(evidence_strategy=EvidenceStrategy.CCS),
    )[0]
    cardinality_step = reduced.decoder_spec.steps[-1]

    assert isinstance(cardinality_step, CardinalityDecoderSpec)
    assert len(cardinality_step.constraints) == 3
