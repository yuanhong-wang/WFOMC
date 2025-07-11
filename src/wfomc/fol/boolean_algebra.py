"""
Logic symbols manipulations
"""
from __future__ import annotations

import typing
import sympy
from sympy import Symbol, satisfiable
from sympy.logic import boolalg
from typing import Dict, Iterable

if typing.TYPE_CHECKING:
    from .syntax import AtomicFormula


Symbol = Symbol


sym2atom: Dict[Symbol, AtomicFormula] = dict()


def get_symbol(atom: AtomicFormula) -> Symbol:
    s = Symbol(str(atom))
    sym2atom[s] = atom
    return s

def get_atom(symbol: Symbol) -> AtomicFormula:
    if symbol not in sym2atom:
        raise RuntimeError(
            "Symbol %s not found", symbol
        )
    return sym2atom.get(symbol)

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
    return frozenset(atoms)

def get_models(expr: Expr) -> Iterable[dict[Symbol, bool]]:
    yield from satisfiable(expr, all_models=True)

def substitute(expr: Expr, mapping: dict[Symbol, Symbol]) -> Expr:
    with sympy.evaluate(False):
        return expr.subs(mapping)

def simplify(symbol: Symbol) -> Symbol:
    return symbol.simplify()
