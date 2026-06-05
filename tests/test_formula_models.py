"""Regression tests for QFFormula.models() on atomless / tautology formulas."""
from wfomc.fol.syntax import top, bot, Pred, X


def test_top_formula_has_single_empty_model():
    # A tautology has no atoms, so it has exactly one model: the empty
    # assignment. Previously models() raised because the SAT backend returned a
    # boolean constant (True / None) that is not a registered atom.
    assert list(top.models()) == [frozenset()]


def test_unsatisfiable_formula_has_no_models():
    P = Pred("P", 1)
    assert list((P(X) & ~P(X)).models()) == []


def test_atomic_formula_models_unchanged():
    P = Pred("P", 1)
    assert list(P(X).models()) == [frozenset({P(X)})]
    assert set((P(X) | ~P(X)).models()) == {
        frozenset({P(X)}),
        frozenset({~P(X)}),
    }
