from abc import ABC
from collections import defaultdict
from enum import Enum
from functools import reduce
from typing import Callable, Iterable
from loguru import logger
from dataclasses import dataclass

from flint import fmpq_mpoly

from wfomc.fol import AUXILIARY_PRED_NAME, AtomicFormula, Const, Pred, X, QFFormula, top
from wfomc.fol import exactly_one_qf, new_predicate
from wfomc.utils import Expr, MultinomialCoefficients, RingElement, create_vars
from wfomc.utils.polynomial_flint import filter_poly
from wfomc.utils.polynomial_flint import to_symexpr


class Constraint(ABC):
    pass


@dataclass(frozen=True)
class TreeConstraint(Constraint):
    pred: Pred

    def __str__(self):
        return "Tree({})".format(self.pred)

    def __repr__(self):
        return str(self)


class CardinalityConstraint(Constraint):
    def __init__(self, constraints: list[tuple[dict[Pred, float], str, float]] = None):
        self.constraints: list[tuple[dict[Pred, float], str, float]] = constraints
        if self.constraints is None:
            self.constraints = list()

        self.preds: list[Pred] = list()
        if self.constraints is not None:
            for constraint in self.constraints:
                self.preds = list(set(self.preds).union(constraint[0].keys()))

        self.gen_vars: list[Expr] = list()
        self.validator: str = ""

    def empty(self) -> bool:
        return len(self.constraints) == 0

    def transform_weighting(self, get_weight: Callable[[Pred], tuple[Expr, Expr]]) \
            -> dict[Pred, tuple[Expr, Expr]]:
        new_weights: dict[Pred, tuple[Expr, Expr]] = {}
        gen_vars = create_vars('x', len(self.preds))
        for sym, pred in zip(gen_vars, self.preds):
            weight = get_weight(pred)
            new_weights[pred] = (weight[0] * sym, weight[1])
            self.gen_vars.append(sym)
        return new_weights

    def decode_poly(self, poly: fmpq_mpoly) -> RingElement:
        res = filter_poly(poly, self.gen_vars, self.valid)
        return res

    def valid(self, degrees: list[int]) -> bool:
        kwargs = zip((pred.name for pred in self.preds), degrees)
        return eval(self.validator.format(**dict(kwargs)))

    def extend_simple_constraints(self, ccs: list[tuple[Pred, str, int]]):
        for pred, comp, card in ccs:
            self.add_simple_constraint(pred, comp, card)

    def add_simple_constraint(self, pred: Pred, comp: str, card: int):
        """
        Add a constraint of the form |pred| comp card
        """
        self.constraints.append(({pred: 1}, comp, card))
        self.preds = list(set(self.preds).union({pred}))

    def add(self, expr: dict[Pred, float], comp: str, param: float):
        self.constraints.append((expr, comp, param))
        self.preds = list(set(self.preds).union(expr.keys()))

    def build(self):
        validator_list: list[str] = []
        for expr, comp, param in self.constraints:
            single_validator = []
            for pred, coef in expr.items():
                single_validator.append(f'{coef} * {{{pred.name}}}')
            single_validator = ' + '.join(single_validator)
            if comp == '=':
                comp = '=='
            validator_list.append(f'{single_validator} {comp} {param}')
        self.validator = ' and '.join(validator_list)
        logger.info('cardinality validator: \n{}', self.validator)

    def __str__(self):
        s = ''
        for expr, comp, param in self.constraints:
            s += ' + '.join(f'{coef} * |{pred.name}|' for pred, coef in expr.items())
            s += ' {} {}'.format(comp, param)
            s += '\n'
        return s

    def __repr__(self):
        return str(self)

