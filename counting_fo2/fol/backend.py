"""
Logic symbols manipulations
"""
from __future__ import annotations

import typing
from sympy import Symbol, satisfiable
from sympy.logic import boolalg
from sympy.logic import simplify_logic
from typing import Dict, Iterable

if typing.TYPE_CHECKING:
    from .syntax import AtomicFormula, CNF, QFFormula


Symbol = Symbol


atom2sym: Dict[AtomicFormula, Symbol] = dict()
sym2atom: Dict[Symbol, AtomicFormula] = dict()


def get_symbol(atom: AtomicFormula) -> Symbol:
    if atom in atom2sym:
        return atom2sym.get(atom)
    s = Symbol(str(atom))
    atom2sym[atom] = s
    sym2atom[s] = atom
    return s

def get_atom(symbol: Symbol) -> AtomicFormula:
    if symbol not in sym2atom:
        raise RuntimeError(
            "Symbol %s not found", symbol
        )
    return sym2atom.get(symbol)


def is_positve(symbol: Symbol) -> bool:
    return not symbol.is_Not


def simplify(symbol: Symbol) -> Symbol:
    return symbol.simplify()


Expr = boolalg.Boolean

def Equivalent(*args):
    return boolalg.Equivalent(*args)

def And(*args):
    return boolalg.And(*args)

def Or(*args):
    return boolalg.Or(*args)

def Not(*args):
    return boolalg.Not(*args)

def Implies(*args):
    return boolalg.Implies(*args)

def get_atoms(expr: Expr) -> frozenset[AtomicFormula]:
    atoms = set()
    for symbol in expr.atoms():
        atoms.add(get_atom(symbol))
    return atoms

def get_models(expr: Expr) -> Iterable[dict[Symbol, bool]]:
    yield from satisfiable(expr, all_models=True)

def substitute(expr: Expr, mapping: dict[Symbol, Symbol]) -> Expr:
    return expr.subs(mapping)

def to_cnf(symbol: boolalg.Boolean, simplify=True) -> CNF:
    from .syntax import AtomicFormula, Lit, DisjunctiveClause, CNF

    def to_internal(symbol: boolalg.Boolean) -> QFFormula:
        if symbol.is_Atom:
            return get_atom(symbol)
        elif symbol.is_Not:
            return Lit(to_internal(symbol.args[0]), False)
        elif isinstance(symbol, boolalg.Or):
            args = [to_internal(arg) for arg in symbol.args]
            lits = [Lit(arg) if isinstance(arg, AtomicFormula)
                    else arg for arg in args]
            return DisjunctiveClause(frozenset(lits))
        elif isinstance(symbol, boolalg.And):
            args = [to_internal(arg) for arg in symbol.args]
            clauses = []
            for arg in args:
                if isinstance(arg, AtomicFormula):
                    clauses.append(DisjunctiveClause(
                        frozenset([Lit(arg)])))
                elif isinstance(arg, Lit):
                    clauses.append(DisjunctiveClause(frozenset([arg])))
                else:
                    clauses.append(arg)
            return CNF(frozenset(clauses))

    if simplify:
        symbol = simplify_logic(symbol)
    cnf = boolalg.to_cnf(symbol)
    return CNF.from_formula(to_internal(cnf))
