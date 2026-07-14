from __future__ import annotations

from fractions import Fraction

import pytest

from wfomc.fol import Literal, Predicate, X
from wfomc.evidence import (
    GroundUnaryLiteral,
    UnaryEvidence,
)
from wfomc.evidence.profile import EvidenceProfile, ProfileCapacityConstraint
from wfomc.cell_graph import required_profile_predicates
from wfomc.reduction import build_profile_capacity_constraint


def _literal(name: str, positive: bool = True) -> Literal:
    return Literal(Predicate(name, 1)(X), positive)


def test_partition_groups_duplicate_profiles_deterministically():
    evidence = UnaryEvidence(
        (
            GroundUnaryLiteral("P", "a", True),
            GroundUnaryLiteral("Q", "a", False),
            GroundUnaryLiteral("P", "b", True),
            GroundUnaryLiteral("Q", "b", False),
            GroundUnaryLiteral("Q", "c", True),
        )
    )

    constraint = build_profile_capacity_constraint(
        evidence, domain=frozenset({"a", "b", "c", "d"})
    )

    assert constraint.domain_size == 4
    assert constraint.assignment_count == Fraction(12, 1)
    assert constraint.profiles == (
        EvidenceProfile(frozenset(), 1),
        EvidenceProfile(frozenset({_literal("P"), _literal("Q", False)}), 2),
        EvidenceProfile(frozenset({_literal("Q")}), 1),
    )
    assert required_profile_predicates(constraint) == frozenset(
        (Predicate("P", 1), Predicate("Q", 1))
    )


def test_partition_rejects_conflicting_evidence():
    evidence = UnaryEvidence(
        (
            GroundUnaryLiteral("P", "a", True),
            GroundUnaryLiteral("P", "a", False),
        )
    )

    with pytest.raises(ValueError, match="consistent"):
        build_profile_capacity_constraint(evidence, domain=frozenset({"a"}))


def test_partition_rejects_evidence_outside_domain():
    evidence = UnaryEvidence((GroundUnaryLiteral("P", "missing", True),))

    with pytest.raises(ValueError, match="domain"):
        build_profile_capacity_constraint(evidence, domain=frozenset({"a"}))


def test_profile_capacity_constraint_from_unary_evidence_preserves_partition():
    evidence = UnaryEvidence(
        (
            GroundUnaryLiteral("P", "a", True),
            GroundUnaryLiteral("Q", "a", False),
            GroundUnaryLiteral("P", "b", True),
        )
    )
    domain = frozenset({"a", "b", "c"})

    constraint = build_profile_capacity_constraint(evidence, domain)

    assert len(constraint.profiles) == 3
    assert constraint.domain_size == 3
    assert constraint.assignment_count == 6


def test_profile_capacity_constraint_is_empty_when_no_profiles():
    constraint = ProfileCapacityConstraint(profiles=(), domain_size=0)
    assert constraint.is_empty
    assert constraint.assignment_count == 1


def test_profile_capacity_constraint_is_not_empty_with_profiles():
    profile = EvidenceProfile(frozenset({_literal("P")}), 1)
    constraint = ProfileCapacityConstraint(profiles=(profile,), domain_size=1)
    assert not constraint.is_empty


def test_profile_capacity_constraint_from_empty_evidence_nonempty_domain():
    # Empty evidence over a non-empty domain yields one empty-literal profile
    # (every element takes the empty profile), so the constraint is not empty.
    constraint = build_profile_capacity_constraint(
        UnaryEvidence(), domain=frozenset({"a"})
    )
    assert len(constraint.profiles) == 1
    assert constraint.profiles[0].literals == frozenset()
    assert not constraint.is_empty
    assert constraint.domain_size == 1


def test_profile_sizes_must_sum_to_domain_size():
    with pytest.raises(ValueError, match="sum to the domain size"):
        ProfileCapacityConstraint(
            profiles=(EvidenceProfile(frozenset(), 1),),
            domain_size=2,
        )


def test_profile_capacity_constraint_from_empty_domain_is_empty():
    constraint = build_profile_capacity_constraint(UnaryEvidence(), domain=frozenset())
    assert constraint.is_empty
    assert constraint.profiles == ()
    assert constraint.domain_size == 0
