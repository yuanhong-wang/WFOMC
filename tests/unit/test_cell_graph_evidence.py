"""Cell-graph allocation tests for lifted unary-evidence profiles."""

from collections import defaultdict
from itertools import product

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.cell_graph import (
    Cell,
    CellConfigCoefficientBasis,
    CellEvidenceAllocation,
    build_cell_graphs,
    profile_cell_formulas,
)
from wfomc.evidence.profile import EvidenceProfile, ProfileCapacityConstraint
from wfomc.fol import Formula, Literal, Predicate, X
from wfomc.fol.semantics import evaluate
from wfomc.multinomial import multinomial_coefficient


def _arithmetic() -> ArithmeticContext:
    return ArithmeticContext(ArithmeticBackend.FMPQ)


def _cells(p, q):
    preds = (p, q)
    return tuple(
        Cell(code, preds)
        for code in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        )
    )


def _overlapping_allocation():
    p = Predicate("FactorP", 1)
    q = Predicate("FactorQ", 1)
    constraint = ProfileCapacityConstraint(
        profiles=(
            EvidenceProfile(frozenset({p(X)}), 1),
            EvidenceProfile(frozenset({q(X)}), 2),
        ),
        domain_size=3,
        assignment_count=3,
    )
    cells = _cells(p, q)
    return cells, CellEvidenceAllocation.from_constraint(constraint, cells)


def test_compiles_overlapping_cell_profile_compatibility():
    _cells_, allocation = _overlapping_allocation()

    assert allocation.compatible_evidence_profiles_by_cell == ((), (0,), (1,), (0, 1))
    assert allocation.compatible_cells_by_evidence_profile == ((1, 3), (2, 3))
    assert allocation.compatible_cell_indices == (1, 2, 3)


def test_config_coefficients_match_brute_force_named_assignments():
    cells, allocation = _overlapping_allocation()

    expected = defaultdict(int)
    for assignments in product(
        allocation.compatible_cells_by_evidence_profile[0],
        allocation.compatible_cells_by_evidence_profile[1],
        allocation.compatible_cells_by_evidence_profile[1],
    ):
        config = tuple(assignments.count(idx) for idx in range(len(cells)))
        expected[config] += 1

    arithmetic = _arithmetic()
    actual = dict(allocation.iter_config_coefficients(arithmetic))

    assert actual == {
        config: arithmetic.from_int(coefficient)
        for config, coefficient in expected.items()
    }


def test_relative_config_coefficients_remove_cell_multinomial():
    _cells_, allocation = _overlapping_allocation()

    arithmetic = _arithmetic()
    absolute = dict(allocation.iter_config_coefficients(arithmetic))
    relative = dict(
        allocation.iter_config_coefficients(
            arithmetic,
            CellConfigCoefficientBasis.RELATIVE_TO_CELL_MULTINOMIAL,
        )
    )

    assert relative == {
        config: coefficient
        * arithmetic.from_fraction(1, multinomial_coefficient(config))
        for config, coefficient in absolute.items()
    }


def test_threaded_transitions_normalize_to_config_coefficients():
    cells, allocation = _overlapping_allocation()
    arithmetic = _arithmetic()
    states = {
        ((0,) * len(cells), allocation.initial_remaining_counts()): arithmetic.one()
    }

    for _ in range(3):
        new_states = defaultdict(arithmetic.zero)
        for (config, remaining), value in states.items():
            for cell_idx in allocation.compatible_cell_indices:
                for new_remaining in allocation.next_remaining_counts(
                    remaining, cell_idx
                ):
                    new_config = tuple(
                        count + int(idx == cell_idx)
                        for idx, count in enumerate(config)
                    )
                    new_states[(new_config, new_remaining)] += value
        states = new_states

    zero = (0,) * len(allocation.evidence_profile_sizes)
    threaded = {
        config: allocation.normalize_ordered_evidence_assignments(value, arithmetic)
        for (config, remaining), value in states.items()
        if remaining == zero
    }

    assert threaded == dict(allocation.iter_config_coefficients(arithmetic))


def test_impossible_evidence_profile_has_no_configs():
    p = Predicate("ImpossibleP", 1)
    cells = (Cell((False,), (p,)),)
    constraint = ProfileCapacityConstraint(
        profiles=(EvidenceProfile(frozenset({p(X)}), 1),),
        domain_size=1,
    )
    allocation = CellEvidenceAllocation.from_constraint(constraint, cells)

    assert list(allocation.iter_config_coefficients(_arithmetic())) == []


def test_profile_literals_are_lowered_to_fol_formulas():
    p = Predicate("ProfileP", 1)
    q = Predicate("ProfileQ", 1)
    constraint = ProfileCapacityConstraint(
        profiles=(
            EvidenceProfile(
                frozenset({Literal(p(X)), Literal(q(X), False)}),
                1,
            ),
        ),
        domain_size=1,
    )

    formulas = profile_cell_formulas(constraint)

    assert formulas is not None
    assert isinstance(formulas[0], Formula)
    assert evaluate(formulas[0], {p(X): True, q(X): False})
    assert not evaluate(formulas[0], {p(X): True, q(X): True})


def test_required_unary_predicate_is_assigned_and_weighted_once():
    predicate = Predicate("RequiredT", 1)
    evidence_only = Predicate("RequiredEvidenceOnly", 1)
    arithmetic = _arithmetic()

    data, _graph_weight = next(
        build_cell_graphs(
            predicate(X),
            {
                evidence_only: (
                    arithmetic.from_int(2),
                    arithmetic.from_int(3),
                )
            },
            arithmetic,
            required_unary_preds=frozenset({evidence_only}),
        )
    )

    assert len(data.cells) == 2
    assert set(data.cell_weights) == {
        arithmetic.from_int(2),
        arithmetic.from_int(3),
    }
    assert all(
        factor.total_weight == arithmetic.one()
        for row in data.pair_factors
        for factor in row
    )
