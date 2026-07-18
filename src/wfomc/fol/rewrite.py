"""Generic rewrites for framework FOL formulas."""

from __future__ import annotations

from .context import context_for
from .literals import Literal
from .syntax import (
    And,
    Atom,
    BoolConst,
    CountingQuantifier,
    Eq,
    Formula,
    Iff,
    Implies,
    Not,
    Or,
    Quantifier,
    QuantifierKind,
)


def eliminate_implications(formula: object) -> object:
    if isinstance(formula, Implies):
        ctx = context_for(formula)
        return ctx.disjunction(
            ctx.neg(eliminate_implications(formula.left)),
            eliminate_implications(formula.right),
        )
    if isinstance(formula, Iff):
        ctx = context_for(formula)
        left = eliminate_implications(formula.left)
        right = eliminate_implications(formula.right)
        return ctx.conjunction(
            ctx.disjunction(ctx.neg(left), right),
            ctx.disjunction(ctx.neg(right), left),
        )
    return _map_children(formula, eliminate_implications)


def push_negation(formula: object) -> object:
    if not isinstance(formula, Not):
        return _map_children(formula, push_negation)
    body = formula.body
    ctx = context_for(formula)
    if isinstance(body, Not):
        return push_negation(body.body)
    if isinstance(body, And):
        return ctx.disjunction(*(push_negation(ctx.neg(arg)) for arg in body.args))
    if isinstance(body, Or):
        return ctx.conjunction(*(push_negation(ctx.neg(arg)) for arg in body.args))
    if isinstance(body, Quantifier):
        flipped = (
            QuantifierKind.EXISTS
            if body.kind == QuantifierKind.FORALL
            else QuantifierKind.FORALL
        )
        return ctx._quantifier(flipped, body.variables, push_negation(ctx.neg(body.body)))
    return formula


def to_nnf(formula: object) -> object:
    return push_negation(eliminate_implications(formula))


def substitute(formula: object, mapping: dict[object, object]) -> object:
    if not mapping:
        return formula
    # atom → formula substitution (e.g. {P(a): true})
    if isinstance(formula, Atom) and formula in mapping:
        return mapping[formula]
    if isinstance(formula, Atom):
        ctx = context_for(formula)
        return ctx.atom(
            formula.predicate,
            *tuple(_substitute_term(term, mapping) for term in formula.terms),
        )
    if isinstance(formula, Literal):
        return formula.substitute(mapping)
    if isinstance(formula, Eq):
        ctx = context_for(formula)
        return ctx.eq(
            _substitute_term(formula.left, mapping),
            _substitute_term(formula.right, mapping),
        )
    if isinstance(formula, BoolConst):
        return formula
    if isinstance(formula, Quantifier):
        shadowed = {var: value for var, value in mapping.items() if var not in formula.variables}
        ctx = context_for(formula)
        return ctx._quantifier(
            formula.kind,
            formula.variables,
            substitute(formula.body, shadowed),
        )
    if isinstance(formula, CountingQuantifier):
        shadowed = {
            var: value for var, value in mapping.items() if var != formula.variable
        }
        ctx = context_for(formula)
        return ctx.count(
            formula.variable,
            formula.comparator,
            formula.count,
            substitute(formula.body, shadowed),
        )
    return _map_children(formula, lambda child: substitute(child, mapping))


def flatten_connectives(formula: object) -> object:
    return _map_children(formula, flatten_connectives)


def simplify_boolean(formula: Formula) -> Formula:
    """Simplify quantifier-free Boolean structure without changing FOL meaning.

    Folds ``BoolConst``, removes identities from connectives, and simplifies
    ``Implies``/``Iff``/``Eq`` when either side is a ``BoolConst``.
    """
    ctx = context_for(formula)

    if isinstance(formula, BoolConst):
        return formula

    if isinstance(formula, Not):
        inner = simplify_boolean(formula.body)
        if isinstance(inner, BoolConst):
            return ctx.true() if not inner.value else ctx.false()
        return ctx.neg(inner)

    if isinstance(formula, And):
        args = [simplify_boolean(a) for a in formula.args]
        kept = []
        for a in args:
            if isinstance(a, BoolConst):
                if not a.value:
                    return ctx.false()
                continue
            kept.append(a)
        if not kept:
            return ctx.true()
        if len(kept) == 1:
            return kept[0]
        return ctx.conjunction(*kept)

    if isinstance(formula, Or):
        args = [simplify_boolean(a) for a in formula.args]
        kept = []
        for a in args:
            if isinstance(a, BoolConst):
                if a.value:
                    return ctx.true()
                continue
            kept.append(a)
        if not kept:
            return ctx.false()
        if len(kept) == 1:
            return kept[0]
        return ctx.disjunction(*kept)

    if isinstance(formula, Implies):
        left = simplify_boolean(formula.left)
        right = simplify_boolean(formula.right)
        if isinstance(left, BoolConst):
            return right if left.value else ctx.true()
        if isinstance(right, BoolConst):
            return ctx.neg(left) if not right.value else ctx.true()
        return ctx.implies(left, right)

    if isinstance(formula, Iff):
        left = simplify_boolean(formula.left)
        right = simplify_boolean(formula.right)
        if isinstance(left, BoolConst) and isinstance(right, BoolConst):
            return ctx.true() if left.value == right.value else ctx.false()
        return ctx.iff(left, right)

    if isinstance(formula, Eq):
        if formula.left == formula.right:
            return ctx.true()
        return formula

    return formula


def _map_children(formula: object, transform) -> object:
    if isinstance(formula, Not):
        return context_for(formula).neg(transform(formula.body))
    if isinstance(formula, Literal):
        return transform(formula)
    if isinstance(formula, And):
        return context_for(formula).conjunction(*(transform(arg) for arg in formula.args))
    if isinstance(formula, Or):
        return context_for(formula).disjunction(*(transform(arg) for arg in formula.args))
    if isinstance(formula, Implies):
        return context_for(formula).implies(
            transform(formula.left),
            transform(formula.right),
        )
    if isinstance(formula, Iff):
        return context_for(formula).iff(transform(formula.left), transform(formula.right))
    if isinstance(formula, Quantifier):
        return context_for(formula)._quantifier(
            formula.kind,
            formula.variables,
            transform(formula.body),
        )
    if isinstance(formula, CountingQuantifier):
        return context_for(formula).count(
            formula.variable,
            formula.comparator,
            formula.count,
            transform(formula.body),
        )
    if isinstance(formula, BoolConst):
        return formula
    return formula


def _substitute_term(term: object, mapping: dict[object, object]) -> object:
    if term in mapping:
        return mapping[term]
    return term


__all__ = [
    "eliminate_implications",
    "flatten_connectives",
    "push_negation",
    "simplify_boolean",
    "substitute",
    "to_nnf",
]
