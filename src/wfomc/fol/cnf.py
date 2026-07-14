"""Typed Tseitin CNF encoding for quantifier-free FOL formulas."""

from __future__ import annotations

from dataclasses import dataclass

from wfomc.fol.analysis import is_quantifier_free
from wfomc.fol.syntax import And, Atom, BoolConst, Formula, Iff, Implies, Not, Or


@dataclass(frozen=True)
class TseitinCNF:
    """A model-preserving CNF with explicit original/auxiliary variables."""

    atoms: tuple[Atom, ...]
    atom_to_var: dict[Atom, int]
    clauses: tuple[tuple[int, ...], ...]
    n_vars: int
    auxiliary_vars: frozenset[int]


def encode_tseitin(formula: Formula) -> TseitinCNF:
    """Encode a quantifier-free formula with uniquely defined auxiliaries."""

    if not is_quantifier_free(formula):
        raise ValueError("Tseitin CNF requires a quantifier-free formula")
    original_atoms = ordered_atoms(formula)
    atom_to_var = {atom: index for index, atom in enumerate(original_atoms, start=1)}
    next_var = [len(original_atoms) + 1]
    clauses: list[tuple[int, ...]] = []
    root = _encode(formula, atom_to_var, next_var, clauses)
    clauses.append((root,))
    n_vars = next_var[0] - 1
    return TseitinCNF(
        atoms=original_atoms,
        atom_to_var=atom_to_var,
        clauses=tuple(clauses),
        n_vars=n_vars,
        auxiliary_vars=frozenset(range(len(original_atoms) + 1, n_vars + 1)),
    )


def ordered_atoms(formula: Formula) -> tuple[Atom, ...]:
    """Return structurally unique atoms in deterministic traversal order."""

    result: dict[Atom, None] = {}

    def visit(node: Formula) -> None:
        if isinstance(node, Atom):
            result.setdefault(node, None)
        elif isinstance(node, Not):
            visit(node.body)
        elif isinstance(node, (And, Or)):
            for argument in node.args:
                visit(argument)
        elif isinstance(node, (Implies, Iff)):
            visit(node.left)
            visit(node.right)

    visit(formula)
    return tuple(result)


def _encode(
    formula: Formula,
    atom_to_var: dict[Atom, int],
    next_var: list[int],
    clauses: list[tuple[int, ...]],
) -> int:
    if isinstance(formula, Atom):
        return atom_to_var[formula]
    if isinstance(formula, BoolConst):
        variable = _new_var(next_var)
        clauses.append((variable,) if formula.value else (-variable,))
        return variable
    if isinstance(formula, Not):
        child = _encode(formula.body, atom_to_var, next_var, clauses)
        variable = _new_var(next_var)
        clauses.append((-variable, -child))
        clauses.append((variable, child))
        return variable
    if isinstance(formula, And):
        children = tuple(
            _encode(argument, atom_to_var, next_var, clauses)
            for argument in formula.args
        )
        variable = _new_var(next_var)
        clauses.extend((-variable, child) for child in children)
        clauses.append((variable,) + tuple(-child for child in children))
        return variable
    if isinstance(formula, Or):
        children = tuple(
            _encode(argument, atom_to_var, next_var, clauses)
            for argument in formula.args
        )
        variable = _new_var(next_var)
        clauses.extend((variable, -child) for child in children)
        clauses.append((-variable,) + children)
        return variable
    if isinstance(formula, Implies):
        left = _encode(formula.left, atom_to_var, next_var, clauses)
        right = _encode(formula.right, atom_to_var, next_var, clauses)
        variable = _new_var(next_var)
        clauses.append((-variable, -left, right))
        clauses.append((variable, left))
        clauses.append((variable, -right))
        return variable
    if isinstance(formula, Iff):
        left = _encode(formula.left, atom_to_var, next_var, clauses)
        right = _encode(formula.right, atom_to_var, next_var, clauses)
        variable = _new_var(next_var)
        clauses.append((-variable, -left, right))
        clauses.append((-variable, left, -right))
        clauses.append((variable, left, right))
        clauses.append((variable, -left, -right))
        return variable
    raise TypeError(f"Tseitin CNF does not support {type(formula).__name__}")


def _new_var(next_var: list[int]) -> int:
    variable = next_var[0]
    next_var[0] += 1
    return variable


__all__ = ["TseitinCNF", "encode_tseitin", "ordered_atoms"]
