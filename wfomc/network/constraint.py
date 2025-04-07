from __future__ import annotations

from abc import ABC
from collections import defaultdict
from enum import Enum
from functools import reduce
from typing import Callable
from logzero import logger
from dataclasses import dataclass

from wfomc.fol.syntax import AUXILIARY_PRED_NAME, AtomicFormula, Const, Pred, X, QFFormula, top
from wfomc.fol.utils import exactly_one_qf, new_predicate
from wfomc.utils import Rational
from wfomc.utils.multinomial import MultinomialCoefficients
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
    def __init__(self, constraints: list[tuple[dict[Pred, float], str, float]] = None) -> None:
        self.constraints: list[tuple[dict[Pred, float], str, float]] = constraints
        if self.constraints is None:
            self.constraints = list()

        self.preds: set[Pred] = set()
        if self.constraints is not None:
            for constraint in self.constraints:
                self.preds.update(constraint[0].keys())

    def empty(self) -> bool:
        return len(self.constraints) == 0

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

    def build(self, get_weight: Callable[[Pred], tuple[Rational, Rational]]) -> \
            tuple[
                dict[Pred, tuple[RingElement, RingElement]],
                Callable[[RingElement], RingElement],
            ]:
        """
        Build the cardinality constraint
        :param get_weight: function to get the weight of a predicate

        :return: new weights and a function to decode the polynomial
        """
        new_weights: dict[Pred, tuple[RingElement, RingElement]] = {}

        gen_vars = list()
        validator = ""

        def decode_poly(poly: RingElement) -> RingElement:
            poly = expand(poly)
            coeffs = coeff_dict(poly, gen_vars)
            res = Rational(0, 1)
            for degrees, coeff in coeffs:
                kwargs = zip(
                    (var.name for var in gen_vars),
                    degrees,
                )
                if eval(validator.format(**dict(kwargs))):
                    res = res + coeff
            return res

        USE_CC_ENCODING = False
        if len(self.preds) <= len(self.constraints) and not USE_CC_ENCODING:
            # encoding CC per predicate
            gen_vars = create_vars('x0:{}'.format(
                len(self.preds))
            )
            pred2var: dict[Pred, Symbol] = dict()
            for sym, pred in zip(gen_vars, self.preds):
                weight = get_weight(pred)
                new_weights[pred] = (weight[0] * sym, weight[1])
                pred2var[pred] = sym
            validators = list()
            for expr, comp, param in self.constraints:
                for pred, coef in expr.items():
                    validators.append('{} {} {}'.format(
                        ' + '.join(
                            f'{coef} * {{{pred2var[pred].name}}}'
                            for pred, coef in expr.items()
                        ),
                        comp if comp != '=' else '==',
                        param
                    ))
            validator = ' and '.join(validators)
        else:
            logger.info(
                'More predicates than constraints, using CC encoding'
            )
            gen_vars = create_vars('x0:{}'.format(
                len(self.constraints)
            ))
            validators = list()
            for sym, (expr, comp, param) in zip(gen_vars, self.constraints):
                for pred, coef in expr.items():
                    if pred not in new_weights:
                        weight = get_weight(pred)
                    else:
                        weight = new_weights[pred]
                    new_weights[pred] = (weight[0] * sym ** coef, weight[1])
                validators.append(
                    f'{{{sym.name}}} {comp if comp != "=" else "=="} {param}'
                )
            validator = ' and '.join(validators)

        return new_weights, decode_poly

    def __str__(self):
        s = ''
        for expr, comp, param in self.constraints:
            s += ' + '.join(f'{coef} * |{pred.name}|' for pred, coef in expr.items())
            s += ' {} {}'.format(comp, param)
            s += '\n'
        return s

    def __repr__(self):
        return str(self)


class PartitionConstraint(Constraint):
    def __init__(self, partition: list[tuple[Pred, int]]) -> None:
        self.partition: list[tuple[Pred, int]] = partition

    def __str__(self) -> str:
        return 'Partition({})'.format(self.partition)


class UnaryEvidenceEncoding(Enum):
    # RETAIN = "retain"
    CCS = "ccs"
    PC = "pc"

    def __str__(self):
        return self.value


def organize_evidence(evidence: set[AtomicFormula]) -> dict[Const, set[AtomicFormula]]:
    element2evidence = defaultdict(set)
    for atom in evidence:
        element2evidence[atom.args[0]].add(atom.substitute({atom.args[0]: X}))
    return element2evidence


def unary_evidence_to_ccs(element2evidence: dict[Const, set[AtomicFormula]],
                          domain: set[Const]) \
        -> tuple[QFFormula, list[tuple[Pred, str, int]], int]:
    """
    Convert unary evidence to cardinality constraints
    """
    evi_size = defaultdict(int)
    for _, evidence in element2evidence.items():
        evi_size[frozenset(evidence)] += 1
    # NOTE: empty frozenset represents non unary evidence
    n_elements_with_evidence = sum(evi_size.values())
    if len(domain) - n_elements_with_evidence > 0:
        evi_size[frozenset()] = len(domain) - n_elements_with_evidence
    formula = top
    aux_preds = []
    ccs = list()
    for evidence, size in evi_size.items():
        aux_pred = new_predicate(1, AUXILIARY_PRED_NAME)
        aux_preds.append(aux_pred)
        aux_atom = aux_pred(X)
        if len(evidence) > 0:
            lits = list(
                lit.substitute({lit.args[0]: X}) for lit in evidence
            )
            formula = formula & (
                aux_atom.implies(reduce(lambda x, y: x & y, lits))
            )
        ccs.append((aux_pred, '=', size))
    ns = tuple(n for _, _, n in ccs)
    n_sum = sum(ns)
    repeat_factor = (
        MultinomialCoefficients.coef((n_sum, len(domain) - n_sum))
        * MultinomialCoefficients.coef(ns)
    )
    formula = formula & exactly_one_qf(aux_preds)
    return formula, ccs, repeat_factor


def unary_evidence_to_pc(element2evidence: dict[Const, set[AtomicFormula]],
                         domain: set[Const]) \
        -> tuple[QFFormula, PartitionConstraint]:
    """
    Convert unary evidence to partition constraint
    """
    evi_size = defaultdict(int)
    for _, evidence in element2evidence.items():
        evi_size[frozenset(evidence)] += 1
    # NOTE: empty frozenset represents non unary evidence
    n_elements_with_evidence = sum(evi_size.values())
    if len(domain) - n_elements_with_evidence > 0:
        evi_size[frozenset()] = len(domain) - n_elements_with_evidence
    formula = top
    aux_preds = []
    partition = list()
    for evidence, size in evi_size.items():
        aux_pred = new_predicate(1, AUXILIARY_PRED_NAME)
        aux_preds.append(aux_pred)
        aux_atom = aux_pred(X)
        if len(evidence) > 0:
            lits = list(
                lit.substitute({lit.args[0]: X}) for lit in evidence
            )
            formula = formula & (
                aux_atom.implies(reduce(lambda x, y: x & y, lits))
            )
        partition.append((aux_pred, size))
    formula = formula & exactly_one_qf(aux_preds)
    return formula, PartitionConstraint(partition)
