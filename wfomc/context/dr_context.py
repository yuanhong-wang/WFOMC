from __future__ import annotations
from logzero import logger
from wfomc.fol.sc2 import SC2
from wfomc.fol.utils import new_predicate, convert_counting_formula
from wfomc.fol.syntax import AUXILIARY_PRED_NAME, X, Y, AtomicFormula, Const, Pred, QFFormula, top, a, b
from wfomc.network.constraint import CardinalityConstraint
from wfomc.fol.syntax import *
from wfomc.problems import WFOMCProblem
from wfomc.fol.syntax import AUXILIARY_PRED_NAME, SKOLEM_PRED_NAME
from wfomc.utils.third_typing import RingElement, Rational
from itertools import product


class DRWFOMCContext(object):
    def __init__(self, problem: WFOMCProblem):
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
        # for handling linear order axiom
        if problem.contain_linear_order_axiom():
            self.leq_pred: Pred = Pred('LEQ', 2)
        else:
            self.leq_pred: Pred = None

        self.uni_formula = []  # The following class attributes are the differences from another context file
        self.ext_preds = []
        self.cnt_preds = []
        self.cnt_params = []  # k  (int)

        self.remainder = []  # r       (int)
        self.mod_preds_index = []
        self.exist_mod = False
        self._preprocess()
        self.c_type_shape = tuple()
        self.build_c_type_shape()

        self.binary_evidence = []
        self.get_binary_evidence()

        self.card_preds = []
        self.card_ccs = []
        self.card_vars = []
        self.build_cardinality_constraints()


    def build_c_type_shape(self):
        self.c_type_shape =  list(2 for _ in self.ext_preds)
        for idx, k in enumerate(self.cnt_params):
            if idx in self.mod_preds_index: # This quantifier is of the type ∃_{r mod k}
                self.c_type_shape.append(k) # 0 to k-1
            else: # 是∃_{k}
                self.c_type_shape.append(k + 1)  # 0 to k

    def _preprocess(self):
        """
        Preprocess the logical formula, convert it into a processable quantified free formuala, and introduce auxiliary predicates.
        """
        self.uni_formula = self.sentence.uni_formula
        while not isinstance(self.uni_formula, QFFormula):
            self.uni_formula = self.uni_formula.quantified_formula
        ext_formulas = self.sentence.ext_formulas
        cnt_formulas = self.sentence.cnt_formulas

        # add auxiliary predicates for existential and counting quantified formulas
        for formula in ext_formulas:
            # NOTE: assume all existential formulas are of the form VxEy
            qf_formula = formula.quantified_formula.quantified_formula
            aux_pred = new_predicate(2, AUXILIARY_PRED_NAME)
            self.uni_formula = self.uni_formula & qf_formula.equivalent(aux_pred(X, Y))
            self.ext_preds.append(aux_pred)

        for idx, formula in enumerate(cnt_formulas):
            qscope = formula.quantified_formula.quantifier_scope
            cnt_param_raw = qscope.count_param  # It could be an int or it could be (r, k)

            if qscope.comparator == 'mod':
                self.exist_mod = True
                self.mod_preds_index.append(idx)

                # cnt_param_raw 是 (r, k)
                r, k = cnt_param_raw
                self.remainder.append(r)
                self.cnt_params.append(k)  # Just put k into cnt_params
            else:
                self.remainder.append(None)
                self.cnt_params.append(cnt_param_raw)

            aux_pred = new_predicate(2, AUXILIARY_PRED_NAME)
            qf_formula = formula.quantified_formula.quantified_formula
            self.uni_formula &= qf_formula.equivalent(aux_pred(X, Y))
            self.cnt_preds.append(aux_pred)

    def build_cardinality_constraints(self):  # this code is under construction
        if self.contain_cardinality_constraint():
            pred2var = dict((pred, var) for var, pred in self.cardinality_constraint.var2pred.items())
            constraints = self.cardinality_constraint.constraints
            for constraint in constraints:
                coeffs, comp, param = constraint
                assert len(coeffs) == 1 and comp == '='
                param = int(param)
                pred, coef = next(iter(coeffs.items()))
                self.card_preds.append(pred)
                assert coef == 1
                self.card_ccs.append(param)
                self.card_vars.append(pred2var[pred])

    def contain_cardinality_constraint(self) -> bool:
        return self.cardinality_constraint is not None and \
            not self.cardinality_constraint.empty()

    def get_binary_evidence(self):
        ext_atoms = list(
            ((~pred(a, b), ~pred(b, a)),
             (~pred(a, b), pred(b, a)),
             (pred(a, b), ~pred(b, a)),
             (pred(a, b), pred(b, a)))
            for pred in self.ext_preds[::-1])
        cnt_atoms = list(
            ((~pred(a, b), ~pred(b, a)),
             (~pred(a, b), pred(b, a)),
             (pred(a, b), ~pred(b, a)),
             (pred(a, b), pred(b, a)))
            for pred in self.cnt_preds[::-1])
        for atoms in product(*cnt_atoms, *ext_atoms):
            self.binary_evidence.append(frozenset(sum(atoms, start=())))

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return default, default
