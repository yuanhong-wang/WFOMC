import pytest

from wfomc.context import EvidenceProfile, UnaryEvidencePartition
from wfomc.fol import Const, Pred, X
from wfomc.utils import Rational


def test_groups_duplicate_profiles_and_negative_literals_deterministically():
    p = Pred("ConstraintP", 1)
    q = Pred("ConstraintQ", 1)
    a = Const("a")
    b = Const("b")
    c = Const("c")
    d = Const("d")

    partition = UnaryEvidencePartition.from_evidence(
        {p(a), p(b), ~q(a), ~q(b), q(c)},
        {d, c, b, a},
    )

    assert partition.evidence_profiles == (
        EvidenceProfile(frozenset(), 1),
        EvidenceProfile(frozenset({p(X), ~q(X)}), 2),
        EvidenceProfile(frozenset({q(X)}), 1),
    )
    assert partition.predicates == frozenset({p, q})
    assert not partition.covers_all_elements
    assert partition.evidence_assignment_count == Rational(12, 1)


def test_empty_evidence_creates_one_unrestricted_evidence_profile():
    domain = {Const("a"), Const("b")}

    partition = UnaryEvidencePartition.from_evidence(set(), domain)

    assert partition.evidence_profiles == (EvidenceProfile(frozenset(), 2),)
    assert not partition.covers_all_elements
    assert partition.coverage_formula().preds() == frozenset()


def test_coverage_formula_contains_only_nonempty_evidence_profile_predicates():
    p = Pred("CoverageP", 1)
    q = Pred("CoverageQ", 1)
    a = Const("a")
    b = Const("b")

    partition = UnaryEvidencePartition.from_evidence({p(a), q(b)}, {a, b})

    assert partition.covers_all_elements
    assert partition.coverage_formula().preds() == frozenset({p, q})


def test_evidence_profile_sizes_must_sum_to_domain_size():
    with pytest.raises(ValueError, match="sum to the domain size"):
        UnaryEvidencePartition((EvidenceProfile(frozenset(), 1),), 2)


def test_evidence_must_be_ground_unary_and_consistent():
    p = Pred("EvidenceP", 1)
    r = Pred("EvidenceR", 2)
    a = Const("a")
    b = Const("b")

    with pytest.raises(ValueError, match="unary"):
        UnaryEvidencePartition.from_evidence({r(a, b)}, {a, b})
    with pytest.raises(ValueError, match="consistent"):
        UnaryEvidencePartition.from_evidence({p(a), ~p(a)}, {a})
