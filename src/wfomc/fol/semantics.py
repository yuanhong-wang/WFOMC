"""Truth, model, SAT, and CNF semantics for typed FOL formulas.

All semantics functions require quantifier-free formulas; quantified
formulas must be reduced (e.g. via CellGraph materialisation) before
calling into this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import product

from wfomc.fol.analysis import is_quantifier_free
from wfomc.fol.syntax import And, Atom, BoolConst, Eq, Formula, Iff, Implies, Not, Or
from wfomc.fol.literals import Literal


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _require_quantifier_free(formula: Formula) -> None:
    if not is_quantifier_free(formula):
        raise ValueError(
            "Formula semantics are only defined for quantifier-free formulas"
        )


# ---------------------------------------------------------------------------
# Atom collection
# ---------------------------------------------------------------------------


def _atoms_list(formula: Formula) -> tuple[Atom, ...]:
    seen: dict[int, Atom] = {}
    _collect(formula, seen)
    return tuple(seen.values())


def _collect(formula: Formula, acc: dict[int, Atom]) -> None:
    if isinstance(formula, Atom):
        if id(formula) not in acc:
            acc[id(formula)] = formula
    elif isinstance(formula, (Not,)):
        _collect(formula.body, acc)
    elif isinstance(formula, (And, Or)):
        for arg in formula.args:
            _collect(arg, acc)
    elif isinstance(formula, (Implies, Iff)):
        _collect(formula.left, acc)
        _collect(formula.right, acc)


# ---------------------------------------------------------------------------
# Formula evaluation (truth-table)
# ---------------------------------------------------------------------------


def evaluate(formula: Formula, assignment: dict[Atom, bool]) -> bool:
    _require_quantifier_free(formula)
    return _evaluate(formula, assignment)


def _evaluate(formula: Formula, assignment: dict[Atom, bool]) -> bool:
    if isinstance(formula, Atom):
        return assignment.get(formula, False)
    if isinstance(formula, BoolConst):
        return formula.value
    if isinstance(formula, Not):
        return not _evaluate(formula.body, assignment)
    if isinstance(formula, And):
        return all(_evaluate(a, assignment) for a in formula.args)
    if isinstance(formula, Or):
        return any(_evaluate(a, assignment) for a in formula.args)
    if isinstance(formula, Implies):
        return (not _evaluate(formula.left, assignment)) or _evaluate(
            formula.right, assignment
        )
    if isinstance(formula, Iff):
        return _evaluate(formula.left, assignment) == _evaluate(
            formula.right, assignment
        )
    if isinstance(formula, Eq):
        return _evaluate_eq(formula, assignment)
    return True


def _evaluate_eq(eq: Eq, assignment: dict[Atom, bool]) -> bool:
    for atom, value in assignment.items():
        if isinstance(atom, Eq) and atom.left == eq.left and atom.right == eq.right:
            return value
    return eq.left == eq.right


# ---------------------------------------------------------------------------
# Satisfiability
# ---------------------------------------------------------------------------


def is_satisfiable(formula: Formula) -> bool:
    _require_quantifier_free(formula)
    try:
        next(models(formula, max_models=1))
        return True
    except StopIteration:
        return False


# ---------------------------------------------------------------------------
# Model enumeration
# ---------------------------------------------------------------------------


def models(
    formula: Formula,
    atoms: Iterable[Atom] | None = None,
    *,
    max_models: int | None = None,
) -> Iterator[dict[Atom, bool]]:
    """Enumerate models as ``{Atom: bool}`` dictionaries."""
    _require_quantifier_free(formula)
    atom_list = _atoms_list(formula) if atoms is None else tuple(atoms)
    if not atom_list:
        if _evaluate(formula, {}):
            yield {}
        return
    count = 0
    for values in product((False, True), repeat=len(atom_list)):
        assignment = dict(zip(atom_list, values))
        if _evaluate(formula, assignment):
            yield assignment
            count += 1
            if max_models is not None and count >= max_models:
                break


def model_literals(
    formula: Formula,
    atom_universe: Iterable[Atom] | None = None,
    *,
    max_models: int | None = None,
) -> Iterator[frozenset[Literal]]:
    """Enumerate models as ``frozenset[Literal]``."""
    _require_quantifier_free(formula)
    atom_list = _atoms_list(formula) if atom_universe is None else tuple(atom_universe)
    if not atom_list:
        if _evaluate(formula, {}):
            yield frozenset()
        return
    yield from _brute_model_literals(formula, atom_list, max_models)


def _brute_model_literals(
    formula: Formula,
    atom_list: tuple[Atom, ...],
    max_models: int | None,
) -> Iterator[frozenset[Literal]]:
    count = 0
    for values in product((False, True), repeat=len(atom_list)):
        assignment = dict(zip(atom_list, values))
        if _evaluate(formula, assignment):
            yield frozenset(Literal(a, v) for a, v in assignment.items())
            count += 1
            if max_models is not None and count >= max_models:
                break


__all__ = [
    "evaluate",
    "is_satisfiable",
    "model_literals",
    "models",
]
