from __future__ import annotations
from logzero import logger
from math import comb
from functools import reduce
from wfomc.fol.syntax import *
from wfomc.fol.utils import convert_counting_formula, new_predicate
from wfomc.network.constraint import CardinalityConstraint
from wfomc.utils.polynomial import Rational
from wfomc.context import WFOMCContext


class WFOMCContextNewEncoding(WFOMCContext):

    def __init__(self, problem: WFOMCProblem, **kwargs):
        super().__init__(
            problem, **kwargs
        )

    def _build(self):
        self.formula = self.sentence.uni_formula
        while not isinstance(self.formula, QFFormula):
            self.formula = self.formula.quantified_formula

        ext_formulas_to_skolemize = list(
            self.sentence.ext_formulas
        )
        if self.sentence.contain_counting_quantifier():
            logger.info("Translating SC2 to SNF using the NEW encoding logic.")
            if not self.contain_cardinality_constraint():
                self.cardinality_constraint = CardinalityConstraint()

            for (
                cnt_formula
            ) in self.sentence.cnt_formulas:
                is_unary_counting = not isinstance(
                    cnt_formula.quantified_formula, QuantifiedFormula
                )
                if is_unary_counting:
                    logger.info(
                        f"Handling unary counting formula (direct conversion): {cnt_formula}"
                    )
                    uni_formula, ext_formulas, card_constraint, repeat_factor = (
                        convert_counting_formula(cnt_formula, self.domain)
                    )
                    self.cardinality_constraint.add_simple_constraint(
                        *card_constraint)
                else:
                    logger.info(
                        f"Handling binary counting formula with NEW encoding: {cnt_formula}"
                    )
                    (
                        uni_formula_old,
                        ext_formulas_from_cnt,
                        card_constraint,
                        repeat_factor,
                    ) = convert_counting_formula(cnt_formula, self.domain)

                    k = (
                        cnt_formula.quantified_formula.quantifier_scope.count_param
                    )
                    skolem_preds = []
                    skolem_axioms = top
                    for (
                        ext_formula
                    ) in ext_formulas_from_cnt:
                        inner_formula = (
                            ext_formula.quantified_formula.quantified_formula
                        )
                        skolem_pred = new_predicate(
                            1, SKOLEM_PRED_NAME
                        )
                        skolem_preds.append(skolem_pred)
                        X = Var("X")
                        axiom_body = (
                            skolem_pred(X) | ~inner_formula
                        )
                        skolem_axioms &= axiom_body
                        self.weights[skolem_pred] = (
                            Rational(1, 1),
                            Rational(-1, 1),
                        )
                    c_preds = [new_predicate(1, f"C_{j}_")
                               for j in range(k + 1)]
                    gamma_c_body = top
                    X = Var("X")
                    for j in range(k + 1):
                        true_atoms = [skolem_preds[h](X) for h in range(j)]
                        false_atoms = [~skolem_preds[h]
                                       (X) for h in range(j, k)]
                        conj_true_A = reduce(
                            lambda f1, f2: f1 & f2, true_atoms, top)
                        conj_false_A = reduce(
                            lambda f1, f2: f1 & f2, false_atoms, top)
                        definition = conj_true_A & conj_false_A
                        gamma_c_body &= c_preds[j](X).equivalent(
                            definition
                        )
                    disjuncts = [
                        p(X) for p in c_preds
                    ]
                    final_disjunction_body = reduce(
                        lambda f1, f2: f1 | f2, disjuncts, bot
                    )
                    self.formula &= (
                        uni_formula_old
                        & skolem_axioms
                        & gamma_c_body
                        & final_disjunction_body
                    )
                    for j in range(k + 1):
                        weight = Rational(comb(k, j), 1)
                        neg_weight = Rational(1, 1)
                        self.weights[c_preds[j]] = (weight, neg_weight)
                    self.cardinality_constraint.add_simple_constraint(
                        *card_constraint)
                    self.repeat_factor *= repeat_factor
        if self.contain_cardinality_constraint():
            self.cardinality_constraint.build()
        for ext_formula in ext_formulas_to_skolemize:
            self.formula &= self._skolemize_one_formula(ext_formula)

        if self.contain_cardinality_constraint():
            self.weights.update(
                self.cardinality_constraint.transform_weighting(
                    self.get_weight)
            )

        if self.problem.contain_linear_order_axiom():
            self.leq_pred = Pred("LEQ", 2)
