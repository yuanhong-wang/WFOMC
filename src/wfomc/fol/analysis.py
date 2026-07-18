"""Cached analysis helpers for framework FOL formulas."""

from __future__ import annotations

from .syntax import (
    Atom,
    CountingQuantifier,
    Eq,
    Formula,
    Quantifier,
    formula_children,
)
from .syntax import Constant, Variable


def predicates(formula: object) -> frozenset[object]:
    return _cached(formula, "predicates", lambda: frozenset(_predicates(formula)))


def constants(formula: object) -> frozenset[object]:
    return _cached(formula, "constants", lambda: frozenset(_constants(formula)))


def free_vars(formula: object) -> frozenset[object]:
    return _cached(
        formula,
        "free_vars",
        lambda: frozenset(_free_vars(formula, bound=frozenset())),
    )


def variables(formula: object) -> frozenset[object]:
    return free_vars(formula)


def _cached(formula: object, key: str, compute):
    if not isinstance(formula, Formula):
        return compute()
    return formula.cached_analysis(key, compute)


def _predicates(formula: object):
    if isinstance(formula, Atom):
        yield formula.predicate
        return
    if isinstance(formula, Formula):
        for child in formula_children(formula):
            yield from _predicates(child)
    return


def _constants(formula: object):
    if isinstance(formula, Atom):
        for term in formula.terms:
            yield from _term_constants(term)
        return
    if isinstance(formula, Eq):
        yield from _term_constants(formula.left)
        yield from _term_constants(formula.right)
        return
    if isinstance(formula, Formula):
        for child in formula_children(formula):
            yield from _constants(child)
    return


def _free_vars(formula: object, *, bound: frozenset[object]):
    if isinstance(formula, Atom):
        for term in formula.terms:
            for variable in _term_variables(term):
                if variable not in bound:
                    yield variable
        return
    if isinstance(formula, Eq):
        for term in (formula.left, formula.right):
            for variable in _term_variables(term):
                if variable not in bound:
                    yield variable
        return
    if isinstance(formula, Quantifier):
        yield from _free_vars(
            formula.body,
            bound=bound | frozenset(formula.variables),
        )
        return
    if isinstance(formula, CountingQuantifier):
        yield from _free_vars(
            formula.body,
            bound=bound | frozenset({formula.variable}),
        )
        return
    if isinstance(formula, Formula):
        for child in formula_children(formula):
            yield from _free_vars(child, bound=bound)
    return


def _term_variables(term: object):
    if isinstance(term, Variable) or term.__class__.__name__ == "Var":
        yield term
        return


def _term_constants(term: object):
    if isinstance(term, Constant) or term.__class__.__name__ == "Const":
        yield term
        return


def is_quantifier_free(formula: object) -> bool:
    if not isinstance(formula, Formula):
        return True
    if isinstance(formula, (Quantifier, CountingQuantifier)):
        return False
    for child in formula_children(formula):
        if not is_quantifier_free(child):
            return False
    return True


def atoms(formula: object) -> frozenset[Atom]:
    result: set[Atom] = set()
    _collect_atoms(formula, result)
    return frozenset(result)


def _collect_atoms(formula: object, acc: set[Atom]) -> None:
    if isinstance(formula, Atom):
        acc.add(formula)
    elif isinstance(formula, Formula):
        for child in formula_children(formula):
            _collect_atoms(child, acc)


__all__ = [
    "atoms",
    "constants",
    "free_vars",
    "is_quantifier_free",
    "predicates",
    "variables",
]
