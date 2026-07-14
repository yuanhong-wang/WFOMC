"""Typed FOL transformer for the framework parser."""

from __future__ import annotations

from functools import reduce
from itertools import combinations

from lark import Lark, Transformer

from wfomc.evidence import GroundUnaryLiteral, UnaryEvidence
from wfomc.fol import (
    FOLContext,
    Formula,
    FormulaKind,
    Predicate,
)

from wfomc.parser.grammar.fol import FORMULA_GRAMMAR


class FormulaTransformer(Transformer):
    def __init__(self, *, context: FOLContext | None = None, **kwargs):
        super().__init__(**kwargs)
        self.context = context if context is not None else FOLContext()
        self.name2pred: dict[str, Predicate] = {}

    def constant(self, args):
        return self.context.constant(args[0].value)

    def variable(self, args):
        return self.context.variable(args[0].value)

    def terms(self, args):
        return list(args)

    def predicate(self, args):
        pred_name = args[0].value
        if pred_name == "PRED":
            pred_name = "PRED1"
        return pred_name

    def atomic_ffl(self, args):
        pred_name = args[0]
        terms = args[2] if len(args) == 4 else []
        if terms is None:
            terms = []
        predicate = self.context.predicate(pred_name, len(terms))
        self.name2pred[pred_name] = predicate
        return self.context.atom(predicate, *terms)

    def nullary_atomic(self, args):
        pred_name = args[0]
        predicate = self.context.predicate(pred_name, 0)
        self.name2pred[pred_name] = predicate
        return self.context.atom(predicate)

    def parenthesis(self, args):
        return args[1]

    def parenthesized_body(self, args):
        return args[1]

    def disjunction(self, args):
        return self.context.disjunction(args[0], args[-1])

    def conjunction(self, args):
        return self.context.conjunction(args[0], args[-1])

    def implication(self, args):
        return self.context.implies(args[0], args[-1])

    def equivalence(self, args):
        return self.context.iff(args[0], args[-1])

    def negation(self, args):
        return self.context.neg(args[-1])

    def universal_quantifier(self, args):
        return ("forall",)

    def existential_quantifier(self, args):
        return ("exists",)

    def equality(self, args):
        return "="

    def nequality(self, args):
        return "!="

    def le(self, args):
        return "<="

    def ge(self, args):
        return ">="

    def lt(self, args):
        return "<"

    def gt(self, args):
        return ">"

    def count_parameter(self, args):
        param = int(args[0])
        if param < 0:
            raise ValueError("Counting parameter must be non-negative")
        return param

    def bounded_count(self, args):
        comparator, param = args
        return (comparator, param)

    def mod_count(self, args):
        remainder, modulus = args
        return ("mod", (remainder, modulus))

    def counting_quantifier(self, args):
        comparator, param = args[0]
        return ("count", comparator, param)

    def quantifier_variable(self, args):
        qinfo, var = args
        return (*qinfo, var)

    def quantification(self, args):
        qinfo, formula = args
        kind = qinfo[0]
        if kind == "forall":
            return self.context.forall(qinfo[1], formula)
        if kind == "exists":
            return self.context.exists(qinfo[1], formula)
        _kind, comparator, param, variable = qinfo
        return self.context.count(variable, comparator, param, formula)

    def exactlyone(self, args):
        predicate_names = args[1]
        predicates = [self.context.predicate(name, 1) for name in predicate_names]
        self.name2pred.update((predicate.name, predicate) for predicate in predicates)
        x = self.context.variable("X")
        literals = tuple(self.context.atom(predicate, x) for predicate in predicates)
        if len(literals) <= 1:
            body = (
                self.context.disjunction(literals[0], self.context.neg(literals[0]))
                if literals
                else self.context.atom("__true", x)
            )
        else:
            at_least_one = reduce(self.context.disjunction, literals)
            exclusions = tuple(
                self.context.neg(self.context.conjunction(left, right))
                for left, right in combinations(literals, 2)
            )
            body = reduce(self.context.conjunction, (at_least_one, *exclusions))
        return self.context.forall(x, body)

    def predicates(self, args):
        return list(args)

    def unary_evidence(self, args):
        literals = tuple(_ground_unary_literal(arg) for arg in args)
        return UnaryEvidence(
            tuple(
                sorted(
                    set(literals),
                    key=lambda literal: (
                        str(literal.constant),
                        str(literal.predicate),
                        literal.positive,
                    ),
                )
            )
        )


def parse(text: str) -> Formula:
    parser = Lark(FORMULA_GRAMMAR, start="ffl")
    tree = parser.parse(text)
    return FormulaTransformer().transform(tree)


def _ground_unary_literal(formula: object) -> GroundUnaryLiteral:
    positive = True
    if isinstance(formula, Formula) and formula.op == FormulaKind.NOT:
        positive = False
        formula = formula.args[0]
    if not isinstance(formula, Formula) or formula.op != FormulaKind.ATOM:
        raise ValueError(f"Unary evidence must be an atomic literal: {formula}")
    predicate, *terms = formula.args
    if len(terms) != 1:
        raise ValueError(
            f"Unary evidence only supports unary predicates for now: {formula}"
        )
    constant = terms[0]
    if constant.__class__.__name__ != "Constant":
        raise ValueError(f"Unary evidence must be ground: {formula}")
    return GroundUnaryLiteral(predicate, constant, positive)


TypedFOLTransformer = FormulaTransformer


__all__ = ["FormulaTransformer", "TypedFOLTransformer", "parse"]
