"""Pretty-printers for framework FOL syntax nodes."""

from __future__ import annotations

from .syntax import (
    And,
    Atom,
    BoolConst,
    CountingQuantifier,
    Eq,
    Iff,
    Implies,
    Not,
    Or,
    Quantifier,
    QuantifierKind,
)


def format_formula(formula: object) -> str:
    if isinstance(formula, Atom):
        return f"{formula.predicate}({','.join(str(term) for term in formula.terms)})"
    if isinstance(formula, BoolConst):
        return "⊤" if formula.value else "⊥"
    if isinstance(formula, Eq):
        return f"{formula.left} = {formula.right}"
    if isinstance(formula, Not):
        body = format_formula(formula.body)
        if isinstance(formula.body, (Atom, BoolConst, Eq)):
            return f"~{body}"
        return f"~({body})"
    if isinstance(formula, And):
        return _join_formula_args(formula.args, "&")
    if isinstance(formula, Or):
        return _join_formula_args(formula.args, "|")
    if isinstance(formula, Implies):
        return f"({formula.left}) -> ({formula.right})"
    if isinstance(formula, Iff):
        return f"({formula.left}) <-> ({formula.right})"
    if isinstance(formula, Quantifier):
        quantifier = (
            "\\forall"
            if formula.kind == QuantifierKind.FORALL
            else "\\exists"
        )
        variables = " ".join(str(variable) for variable in formula.variables)
        return f"{quantifier} {variables}: {formula.body}"
    if isinstance(formula, CountingQuantifier):
        return (
            f"\\exists_{{{_format_count(formula.comparator, formula.count)}}} "
            f"{formula.variable}: {formula.body}"
        )
    return str(formula)


def _join_formula_args(args: tuple[object, ...], operator: str) -> str:
    return f" {operator} ".join(f"({arg})" for arg in args)


def _format_count(comparator: object, count: object) -> str:
    if comparator == "mod":
        if isinstance(count, tuple) and len(count) == 2:
            return f"{count[0]}mod{count[1]}"
        if hasattr(count, "remainder") and hasattr(count, "modulus"):
            return f"{count.remainder}mod{count.modulus}"
    return f"{comparator}{count}"


__all__ = ["format_formula"]
