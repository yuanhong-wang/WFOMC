"""Public typed first-order logic DSL.

This module is the ergonomic formula-building facade. Data nodes live in
``wfomc.fol.syntax``.
"""

from __future__ import annotations

from .analysis import (
    atoms,
    constants,
    free_vars,
    height,
    is_quantifier_free,
    predicates,
    variables,
)
from .context import FOLContext, context_for, current_context, default_context
from .literals import Literal, positive_atom
from .syntax import (
    And,
    Atom,
    BoolConst,
    CountingQuantifier,
    Eq,
    Formula,
    FormulaKind,
    Iff,
    Implies,
    ModCount,
    Not,
    Or,
    Quantifier,
    QuantifierKind,
    formula_children,
)
from .pretty import format_formula
from .rewrite import (
    eliminate_implications,
    flatten_connectives,
    ground_formula,
    push_negation,
    simplify_boolean,
    substitute,
    to_nnf,
)
from .semantics import (
    evaluate,
    is_satisfiable,
    model_literals,
    models,
)
from .syntax import Constant, Predicate, Sort, Term, Variable


_DEFAULT_CONTEXT = default_context()


def atom(predicate: object, *terms: object) -> Formula:
    return context_for(predicate, *terms).atom(predicate, *terms)


def eq(left: object, right: object) -> Formula:
    return context_for(left, right).eq(left, right)


def neg(formula: object) -> Formula:
    return context_for(formula).neg(formula)


def conjunction(*formulas: object) -> Formula:
    return context_for(*formulas).conjunction(*formulas)


def disjunction(*formulas: object) -> Formula:
    return context_for(*formulas).disjunction(*formulas)


def implies(left: object, right: object) -> Formula:
    return context_for(left, right).implies(left, right)


def iff(left: object, right: object) -> Formula:
    return context_for(left, right).iff(left, right)


def forall(variables: object, body: object) -> Formula:
    return context_for(variables, body).forall(variables, body)


def exists(variables: object, body: object) -> Formula:
    return context_for(variables, body).exists(variables, body)


def count(
    variable: object, comparator: str, count_value: object, body: object
) -> Formula:
    return context_for(variable, body).count(variable, comparator, count_value, body)


def true() -> Formula:
    return _DEFAULT_CONTEXT.true()


def false() -> Formula:
    return _DEFAULT_CONTEXT.false()


def children(node: object) -> tuple[object, ...]:
    return formula_children(node)


def walk(node: object):
    yield node
    for child in children(node):
        yield from walk(child)


X = _DEFAULT_CONTEXT.variable("X")
Y = _DEFAULT_CONTEXT.variable("Y")
Z = _DEFAULT_CONTEXT.variable("Z")
U = _DEFAULT_CONTEXT.variable("U")
V = _DEFAULT_CONTEXT.variable("V")
W = _DEFAULT_CONTEXT.variable("W")

a = _DEFAULT_CONTEXT.constant("a")
b = _DEFAULT_CONTEXT.constant("b")
c = _DEFAULT_CONTEXT.constant("c")


__all__ = [
    "And",
    "Atom",
    "BoolConst",
    "Constant",
    "CountingQuantifier",
    "Eq",
    "FOLContext",
    "Formula",
    "FormulaKind",
    "Iff",
    "Implies",
    "Literal",
    "ModCount",
    "Not",
    "Or",
    "Predicate",
    "Quantifier",
    "QuantifierKind",
    "Sort",
    "Term",
    "U",
    "V",
    "Variable",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "atom",
    "atoms",
    "b",
    "c",
    "children",
    "conjunction",
    "constants",
    "context_for",
    "count",
    "current_context",
    "default_context",
    "disjunction",
    "eliminate_implications",
    "eq",
    "evaluate",
    "exists",
    "false",
    "flatten_connectives",
    "forall",
    "format_formula",
    "free_vars",
    "ground_formula",
    "height",
    "iff",
    "implies",
    "is_quantifier_free",
    "is_satisfiable",
    "model_literals",
    "models",
    "neg",
    "positive_atom",
    "predicates",
    "push_negation",
    "simplify_boolean",
    "substitute",
    "to_nnf",
    "true",
    "variables",
    "walk",
]
