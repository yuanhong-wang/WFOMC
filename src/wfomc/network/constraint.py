from __future__ import annotations

from abc import ABC
from typing import Callable
from logzero import logger
from dataclasses import dataclass

from wfomc.fol.syntax import Pred
from wfomc.utils import Rational
from wfomc.utils.polynomial import coeff_dict, create_vars, Symbol, expand
from wfomc.utils.third_typing import RingElement


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

        self.preds: set[Pred] = set()
        if self.constraints is not None:
            for constraint in self.constraints:
                self.preds.update(constraint[0].keys())

        self.gen_vars: list[Symbol]
        self.var2pred: dict[Symbol, Pred] = dict()
        self.validator: str = ""

    def empty(self) -> bool:
        return len(self.constraints) == 0

    def transform_weighting(self, get_weight: Callable[[Pred], tuple[Rational, Rational]]) \
            -> dict[Pred, tuple[Rational, Rational]]:
        new_weights: dict[Pred, tuple[RingElement, RingElement]] = {}
        self.gen_vars = create_vars('x0:{}'.format(
            len(self.preds))
        )
        for sym, pred in zip(self.gen_vars, self.preds):
            weight = get_weight(pred)
            new_weights[pred] = (weight[0] * sym, weight[1])
            self.var2pred[sym] = pred
        return new_weights

    def decode_poly(self, poly: RingElement) -> RingElement:
        poly = expand(poly)
        coeffs = coeff_dict(poly, self.gen_vars)
        # logger.debug('coeffs: %s', list(coeffs))
        res = Rational(0, 1)
        for degrees, coeff in coeffs:
            if self.valid(degrees):
                res += coeff
        return res

    def valid(self, degrees: list[int]) -> bool:
        kwargs = zip((self.var2pred[sym].name for sym in self.gen_vars), degrees)
        return eval(self.validator.format(**dict(kwargs)))

    def extend_simple_constraints(self, ccs: list[tuple[Pred, str, int]]):
        for pred, comp, card in ccs:
            self.add_simple_constraint(pred, comp, card)

    def add_simple_constraint(self, pred: Pred, comp: str, card: int):
        """
        Add a constraint of the form |pred| comp card
        """
        self.constraints.append(({pred: 1}, comp, card))
        self.preds.add(pred)

    def add(self, expr: dict[Pred, float], comp: str, param: float):
        self.constraints.append((expr, comp, param))
        self.preds.update(expr.keys())

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
        logger.info('cardinality validator: \n%s', self.validator)

    def __str__(self):
        s = ''
        for expr, comp, param in self.constraints:
            s += ' + '.join(f'{coef} * |{pred.name}|' for pred, coef in expr.items())
            s += ' {} {}'.format(comp, param)
            s += '\n'
        return s

    def __repr__(self):
        return str(self)
