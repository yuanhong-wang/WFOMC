from __future__ import annotations
from logzero import logger
from counting_fo2.fol.sc2 import SC2
from counting_fo2.fol.utils import new_predicate, convert_counting_formula

from counting_fo2.network.constraint import CardinalityConstraint
from counting_fo2.fol.syntax import *
from counting_fo2.problems import WFOMCSProblem
from counting_fo2.fol.syntax import AUXILIARY_PRED_NAME, SKOLEM_PRED_NAME
from counting_fo2.utils.third_typing import RingElement, Rational


class WFOMCContext(object):
    """
    Context for WFOMC algorithm
    """

    def __init__(self, problem: WFOMCSProblem):
        self.domain: set[Const] = problem.domain
        self.sentence: SC2 = problem.sentence
        self.weights: dict[Pred, tuple[Rational, Rational]] = problem.weights
        self.cardinality_constraint: CardinalityConstraint = problem.cardinality_constraint
        self.repeat_factor = 1

        logger.info('sentence: \n%s', self.sentence)
        logger.info('domain: \n%s', self.domain)
        logger.info('weights:')
        for pred, w in self.weights.items():
            logger.info('%s: %s', pred, w)
        logger.info('cardinality constraint: %s', self.cardinality_constraint)

        self.formula: QFFormula
        self._build()
        logger.info('Skolemized formula for WFOMC: \n%s', self.formula)
        logger.info('weights for WFOMC: \n%s', self.weights)

    def contain_cardinality_constraint(self) -> bool:
        return self.cardinality_constraint is not None and \
            not self.cardinality_constraint.empty()

    def contain_existential_quantifier(self) -> bool:
        return self.sentence.contain_existential_quantifier()

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return (default, default)

    def decode_result(self, res: RingElement):
        if not self.contain_cardinality_constraint():
            return res / self.repeat_factor
        res = self.cardinality_constraint.decode_poly(res)
        return res / self.repeat_factor

    def _skolemize_one_formula(self, formula: QuantifiedFormula) -> QFFormula:
        """
        Only need to deal with \forall X \exists Y: f(X,Y) or \exists X: f(X,Y)
        """
        quantified_formula = formula.quantified_formula
        quantifier_num = 1
        while(not isinstance(quantified_formula, QFFormula)):
            quantified_formula = quantified_formula.quantified_formula
            quantifier_num += 1

        formula: QFFormula = top
        ext_formula = quantified_formula
        if not isinstance(ext_formula, AtomicFormula):
            aux_pred = new_predicate(quantifier_num, AUXILIARY_PRED_NAME)
            aux_atom = aux_pred(X, Y) if quantifier_num == 2 else aux_pred(X)
            formula = formula & (ext_formula.equivalent(aux_atom))
            ext_formula = aux_atom

        if quantifier_num == 2:
            skolem_pred = new_predicate(1, SKOLEM_PRED_NAME)
            skolem_atom = skolem_pred(X)
        elif quantifier_num == 1:
            skolem_pred = new_predicate(0, SKOLEM_PRED_NAME)
            skolem_atom = skolem_pred()
        formula = formula & (skolem_atom | ~ext_formula)
        self.weights[skolem_pred] = (Rational(1, 1), Rational(-1, 1))
        return formula

    def _skolemize(self) -> QFFormula:
        """
        Skolemize the sentence
        """
        formula = self.sentence.uni_formula
        while(not isinstance(formula, QFFormula)):
            formula = formula.quantified_formula
        for ext_formula in self.sentence.ext_formulas:
            formula = formula & self._skolemize_one_formula(ext_formula)
        return formula

    def _build(self):
        self.formula = self.sentence.uni_formula
        while(not isinstance(self.formula, QFFormula)):
            self.formula = self.formula.quantified_formula

        self.ext_formulas = self.sentence.ext_formulas
        if self.sentence.contain_counting_quantifier():
            logger.info('translate SC2 to SNF')
            if not self.contain_cardinality_constraint():
                self.cardinality_constraint = CardinalityConstraint()
            for cnt_formula in self.sentence.cnt_formulas:
                uni_formula, ext_formulas, cardinality_constraint, repeat_factor = \
                    convert_counting_formula(cnt_formula, self.domain)
                self.formula = self.formula & uni_formula
                self.ext_formulas = self.ext_formulas + ext_formulas
                self.cardinality_constraint.add_simple_constraint(*cardinality_constraint)
                self.repeat_factor *= repeat_factor

        if self.contain_cardinality_constraint():
            self.cardinality_constraint.build()

        for ext_formula in self.ext_formulas:
            self.formula = self.formula & self._skolemize_one_formula(ext_formula)

        # self.formula = self.formula.simplify()

        if self.contain_cardinality_constraint():
            self.weights.update(
                self.cardinality_constraint.transform_weighting(
                    self.get_weight,
                )
            )
