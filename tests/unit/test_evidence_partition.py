from __future__ import annotations

import pytest

from wfomc.fol import Literal, Predicate, X
from wfomc.evidence.profile import EvidenceProfile, ProfileCapacityConstraint


def _literal(name: str, positive: bool = True) -> Literal:
    return Literal(Predicate(name, 1)(X), positive)


def test_profile_capacity_constraint_is_empty_when_no_profiles():
    constraint = ProfileCapacityConstraint(profiles=(), domain_size=0)
    assert constraint.is_empty
    assert constraint.assignment_count == 1


def test_profile_capacity_constraint_is_not_empty_with_profiles():
    profile = EvidenceProfile(frozenset({_literal("P")}), 1)
    constraint = ProfileCapacityConstraint(profiles=(profile,), domain_size=1)
    assert not constraint.is_empty


def test_profile_sizes_must_sum_to_domain_size():
    with pytest.raises(ValueError, match="sum to the domain size"):
        ProfileCapacityConstraint(
            profiles=(EvidenceProfile(frozenset(), 1),),
            domain_size=2,
        )
