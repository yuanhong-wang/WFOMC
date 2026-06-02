from collections import defaultdict
from itertools import product

from wfomc.cell_graph import Cell
from wfomc.context import (
    CellConfigCoefficientBasis,
    EvidenceProfile,
    UnaryEvidencePartition,
)
from wfomc.fol import Pred, X
from wfomc.utils import MultinomialCoefficients, Rational


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
    p = Pred("FactorP", 1)
    q = Pred("FactorQ", 1)
    partition = UnaryEvidencePartition(
        (
            EvidenceProfile(frozenset({p(X)}), 1),
            EvidenceProfile(frozenset({q(X)}), 2),
        ),
        3,
    )
    cells = _cells(p, q)
    return cells, partition.compile_for_cells(cells)


def test_compiles_overlapping_cell_profile_compatibility():
    _cells_, allocation = _overlapping_allocation()

    assert allocation.compatible_evidence_profiles_by_cell == (
        (), (0,), (1,), (0, 1)
    )
    assert allocation.compatible_cells_by_evidence_profile == ((1, 3), (2, 3))
    assert allocation.compatible_cell_indices == (1, 2, 3)
    assert allocation.compatibility_pair_count == 4


def test_config_coefficients_match_brute_force_named_assignments():
    cells, allocation = _overlapping_allocation()
    MultinomialCoefficients.setup(3)

    expected = defaultdict(int)
    for assignments in product(
        allocation.compatible_cells_by_evidence_profile[0],
        allocation.compatible_cells_by_evidence_profile[1],
        allocation.compatible_cells_by_evidence_profile[1],
    ):
        config = tuple(assignments.count(idx) for idx in range(len(cells)))
        expected[config] += 1

    actual = dict(allocation.iter_config_coefficients())

    assert actual == {
        config: Rational(coefficient, 1)
        for config, coefficient in expected.items()
    }


def test_relative_config_coefficients_remove_cell_multinomial():
    _cells_, allocation = _overlapping_allocation()
    MultinomialCoefficients.setup(3)

    absolute = dict(allocation.iter_config_coefficients())
    relative = dict(
        allocation.iter_config_coefficients(
            CellConfigCoefficientBasis.RELATIVE_TO_CELL_MULTINOMIAL
        )
    )

    assert relative == {
        config: coefficient * Rational(
            1, MultinomialCoefficients.coef(config)
        )
        for config, coefficient in absolute.items()
    }


def test_threaded_transitions_normalize_to_config_coefficients():
    cells, allocation = _overlapping_allocation()
    MultinomialCoefficients.setup(3)
    states = {
        (
            (0,) * len(cells),
            allocation.initial_remaining_counts(),
        ): Rational(1, 1)
    }

    for _ in range(3):
        new_states = defaultdict(lambda: Rational(0, 1))
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
        config: allocation.normalize_ordered_evidence_assignments(value)
        for (config, remaining), value in states.items()
        if remaining == zero
    }

    assert threaded == dict(allocation.iter_config_coefficients())


def test_impossible_evidence_profile_has_no_configs():
    p = Pred("ImpossibleP", 1)
    cells = (Cell((False,), (p,)),)
    partition = UnaryEvidencePartition(
        (EvidenceProfile(frozenset({p(X)}), 1),),
        1,
    )
    allocation = partition.compile_for_cells(cells)
    MultinomialCoefficients.setup(1)

    assert list(allocation.iter_config_coefficients()) == []
