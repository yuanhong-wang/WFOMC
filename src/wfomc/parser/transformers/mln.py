"""MLN grammar transformer."""

from __future__ import annotations

import math
from fractions import Fraction

from wfomc.fol import (
    Predicate,
    atom,
    conjunction,
    forall,
    iff,
)
from wfomc.evidence import Evidence
from wfomc.problem import Domain, Problem, ProblemInstance

from .wfomcs import ProblemTransformer


AUXILIARY_PREDICATE_PREFIX = "@aux"


class MLNTransformer(ProblemTransformer):
    """Transform an MLN parse tree into the native ``Problem`` IR."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._auxiliary_predicate_index = 0

    def weighting(self, args):
        return float(args[0])

    def rules(self, args):
        weights = []
        formulas = []
        for weight, formula in args:
            weights.append(weight)
            formulas.append(formula)
        return weights, formulas

    def rule(self, args):
        return args[0]

    def hard_rule(self, args):
        return float("inf"), args[0]

    def soft_rule(self, args):
        return args[0], args[1]

    def mln(self, args) -> ProblemInstance:
        rules = args[0]
        _domain_name, domain = args[1]
        cardinality_constraints = args[2]
        unary_evidence = args[3]

        formulas = []
        weights = {}
        for weight, formula in zip(*rules):
            rule_formula = formula
            free_vars = tuple(sorted(rule_formula.free_vars(), key=str))
            if weight != float("inf"):
                aux_pred = self._new_auxiliary_predicate(len(free_vars))
                rule_formula = iff(rule_formula, atom(aux_pred, *free_vars))
                weights[aux_pred] = (_mln_positive_weight(weight), Fraction(1, 1))
            for free_var in reversed(free_vars):
                rule_formula = forall(free_var, rule_formula)
            formulas.append(rule_formula)

        return ProblemInstance(
            problem=Problem(
                sentence=_conjoin_formulas(formulas),
                weights=weights,
                cardinality_constraints=self._cardinality_constraints(
                    cardinality_constraints
                ),
                evidence=Evidence(unary=unary_evidence),
            ),
            domain=Domain(frozenset(domain)),
        )

    def _new_auxiliary_predicate(self, arity: int) -> Predicate:
        predicate = self.context.predicate(
            f"{AUXILIARY_PREDICATE_PREFIX}{self._auxiliary_predicate_index}",
            arity,
        )
        self._auxiliary_predicate_index += 1
        return predicate


def _mln_positive_weight(weight: float) -> Fraction:
    fraction = Fraction(math.exp(weight))
    return Fraction(fraction.numerator, fraction.denominator)


def _conjoin_formulas(formulas: list[object]) -> object:
    if not formulas:
        return atom("__true",)
    return formulas[0] if len(formulas) == 1 else conjunction(
        formulas[0],
        _conjoin_formulas(formulas[1:]),
    )


__all__ = ["MLNTransformer"]
