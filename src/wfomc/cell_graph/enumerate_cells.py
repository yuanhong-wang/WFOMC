"""Enumerate satisfiable cells (one-types) with projected SAT."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

from pysat.solvers import Solver

from wfomc.fol import Predicate
from wfomc.fol.cnf import TseitinCNF

from .data import Cell


def is_satisfiable(cnf: TseitinCNF) -> bool:
    """Return whether a typed CNF has at least one model."""

    with Solver(name="cadical195", bootstrap_with=cnf.clauses) as solver:
        return solver.solve()


def enumerate_cells(
    cnf: TseitinCNF,
    predicates: Sequence[Predicate],
) -> tuple[Cell, ...]:
    """Enumerate satisfying assignments projected to one atom per predicate."""

    predicate_order = tuple(predicates)
    atom_var_by_predicate: dict[Predicate, int] = {}
    for atom, variable in cnf.atom_to_var.items():
        predicate = atom.predicate
        if predicate in atom_var_by_predicate:
            raise ValueError(
                f"Cell CNF contains more than one atom for predicate {predicate}"
            )
        atom_var_by_predicate[predicate] = variable
    present = tuple(
        (index, atom_var_by_predicate[predicate])
        for index, predicate in enumerate(predicate_order)
        if predicate in atom_var_by_predicate
    )
    missing_indices = tuple(
        index
        for index, predicate in enumerate(predicate_order)
        if predicate not in atom_var_by_predicate
    )
    project_vars = tuple(variable for _index, variable in present)
    projected_codes: set[tuple[bool, ...]] = set()
    with Solver(name="cadical195", bootstrap_with=cnf.clauses) as solver:
        if not project_vars:
            if not solver.solve():
                return ()
            projected_codes.add(())
        else:
            while solver.solve():
                positive = {literal for literal in solver.get_model() if literal > 0}
                code = tuple(variable in positive for variable in project_vars)
                projected_codes.add(code)
                solver.add_clause(
                    [
                        -variable if value else variable
                        for variable, value in zip(project_vars, code)
                    ]
                )

    codes: list[tuple[bool, ...]] = []
    for projected_code in sorted(projected_codes):
        for missing_code in product((False, True), repeat=len(missing_indices)):
            code = [False] * len(predicate_order)
            for (index, _variable), value in zip(present, projected_code):
                code[index] = value
            for index, value in zip(missing_indices, missing_code):
                code[index] = value
            codes.append(tuple(code))
    return tuple(Cell(code, predicate_order) for code in codes)


__all__ = ["enumerate_cells", "is_satisfiable"]
