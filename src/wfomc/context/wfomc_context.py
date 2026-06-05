from __future__ import annotations

import math
from copy import deepcopy
from functools import reduce
from math import comb

from loguru import logger

from wfomc.cell_graph import Cell, build_cell_graphs as _build_cell_graphs
from wfomc.fol.sc2 import SC2
from wfomc.fol.syntax import *
from wfomc.fol.utils import new_predicate, tseitin_transform
from wfomc.network import (
    CardinalityConstraint,
)
from .unary_evidence import (
    CellEvidenceAllocation,
    UnaryEvidencePartition,
    UnaryEvidencePlan,
    UnaryEvidenceStrategy,
)
from wfomc.problems import WFOMCProblem
from wfomc.utils import Expr, Rational, RingElement, to_ringelements


class WFOMCContext:
    """
    Context for the lifted WFOMC algorithms.

    Public problem weights remain SymPy expressions. The context converts them
    to FLINT ring elements once preprocessing has finished, and algorithms read
    the converted weights through _get_weight().
    """

    def __init__(self, problem: WFOMCProblem,
                 unary_evidence_strategy: UnaryEvidenceStrategy = UnaryEvidenceStrategy.AUTO):
        self.problem = deepcopy(problem)
        self.domain: set[Const] = self.problem.domain
        self.sentence: SC2 = self.problem.sentence
        self.weights: dict[Pred, tuple[Expr, Expr]] = self.problem.weights
        self._weights: dict[Pred, tuple[RingElement, RingElement]] = dict()
        self.ring_element_one: RingElement = to_ringelements([Rational(1, 1)])[0]
        self.cardinality_constraint: CardinalityConstraint = self.problem.cardinality_constraint
        self.repeat_factor = 1
        self.unary_evidence = self.problem.unary_evidence

        self.unary_evidence_strategy = unary_evidence_strategy
        self.unary_evidence_plan: UnaryEvidencePlan | None = None

        logger.info('sentence: \n{}', self.sentence)
        logger.info('domain: \n{}', self.domain)
        logger.info('weights:')
        for pred, w in self.weights.items():
            logger.info('{}: {}', pred, w)
        logger.info('cardinality constraint: {}', self.cardinality_constraint)

        self.leq_pred: Pred | None = None
        if problem.contain_linear_order_axiom():
            self.leq_pred = Pred('LEQ', 2)

        self.predecessor_preds: dict[int, Pred] | None = None
        self.circular_predecessor_pred: Pred | None = None

        self.formula: QFFormula | None = None

        self._build()

        logger.info('Skolemized formula: \n{}', self.formula)
        logger.info('weights for WFOMC: \n{}', self.weights)
        logger.info('repeat factor: {}', self.repeat_factor)
        logger.info('unary evidence: {}', self.unary_evidence)
        logger.info('unary evidence strategy: {}', self.unary_evidence_strategy)

    def contain_cardinality_constraint(self) -> bool:
        return self.cardinality_constraint is not None and \
            not self.cardinality_constraint.empty()

    def _prepare_cardinality_weights(self) -> None:
        """Attach polynomial weights used to enforce cardinality constraints."""
        if self.contain_cardinality_constraint():
            self.cardinality_constraint.build()
            self.weights.update(
                self.cardinality_constraint.transform_weighting(self.get_weight)
            )

    @property
    def required_unary_preds(self) -> frozenset[Pred]:
        if self.unary_evidence_plan is None:
            return frozenset()
        return self.unary_evidence_plan.required_unary_preds

    @property
    def unary_evidence_partition(self) -> UnaryEvidencePartition | None:
        if self.unary_evidence_plan is None:
            return None
        return self.unary_evidence_plan.partition

    @property
    def uses_lifted_unary_evidence(self) -> bool:
        return (
            self.unary_evidence_plan is not None
            and self.unary_evidence_plan.uses_cell_allocation
        )

    def cell_evidence_allocation(
        self,
        cells: list[Cell] | tuple[Cell, ...],
    ) -> CellEvidenceAllocation | None:
        if self.unary_evidence_plan is None:
            return None
        return self.unary_evidence_plan.compile_for_cells(cells)

    def build_cell_graphs(self, **kwargs):
        """Build cell graphs with all context-owned unary evidence metadata."""
        kwargs.setdefault("required_unary_preds", self.required_unary_preds)
        if self.uses_lifted_unary_evidence:
            kwargs.setdefault(
                "unary_evidence_partition", self.unary_evidence_partition
            )
        return _build_cell_graphs(self.formula, self._get_weight, **kwargs)

    def contain_existential_quantifier(self) -> bool:
        return self.sentence.contain_existential_quantifier()

    def contain_linear_order_axiom(self) -> bool:
        return self.leq_pred is not None

    def get_weight(self, pred: Pred) -> tuple[Expr, Expr]:
        if pred in self.weights:
            return self.weights[pred]
        return (Rational(1, 1), Rational(1, 1))

    def _get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        if pred in self._weights:
            return self._weights[pred]
        return (self.ring_element_one, self.ring_element_one)

    def decode_result(self, res: RingElement) -> RingElement:
        if self.leq_pred is not None:
            res *= math.factorial(len(self.domain))
        res = res / self.repeat_factor
        if res == 0:
            return res
        if self.contain_cardinality_constraint():
            res = self.cardinality_constraint.decode_poly(res)
        return res

    def _skolemize(self) -> None:
        for ext_formula in self.sentence.ext_formulas:
            formula, weights_update = self._skolemize_one_formula(ext_formula)
            self.formula &= formula
            self.weights.update(weights_update)

    def _skolemize_one_formula(self, formula: QuantifiedFormula) -> \
            tuple[QFFormula, dict[Pred, tuple[Expr, Expr]]]:
        ext_formula = formula.quantified_formula
        quantifier_num = 1
        while not isinstance(ext_formula, QFFormula):
            ext_formula = ext_formula.quantified_formula
            quantifier_num += 1

        if quantifier_num == 2:
            skolem_pred = new_predicate(1, SKOLEM_PRED_NAME)
            skolem_atom = skolem_pred(X)
        else:
            skolem_pred = new_predicate(0, SKOLEM_PRED_NAME)
            skolem_atom = skolem_pred()
        return (
            skolem_atom | ~ext_formula,
            {skolem_pred: (Rational(1, 1), Rational(-1, 1))},
        )

    def _transform_forall_existsK(self, formula: QuantifiedFormula) -> \
            tuple[QFFormula, list[QuantifiedFormula], tuple, int]:
        uni_formula = top
        ext_formulas = []

        cnt_quantified_formula = formula.quantified_formula.quantified_formula
        cnt_quantifier = formula.quantified_formula.quantifier_scope
        count_param = int(cnt_quantifier.count_param)

        repeat_factor = (math.factorial(count_param)) ** len(self.domain)

        aux_pred = new_predicate(2, AUXILIARY_PRED_NAME)
        aux_atom = aux_pred(X, Y)
        uni_formula = uni_formula & (
            cnt_quantified_formula.equivalent(aux_atom))

        sub_aux_preds, sub_aux_atoms = [], []
        for i in range(count_param):
            aux_pred_i = new_predicate(2, f'{aux_pred.name}_')
            aux_atom_i = aux_pred_i(X, Y)
            sub_aux_preds.append(aux_pred_i)
            sub_aux_atoms.append(aux_atom_i)
            sub_ext_formula = QuantifiedFormula(Existential(Y), aux_atom_i)
            sub_ext_formula = QuantifiedFormula(Universal(X), sub_ext_formula)
            ext_formulas.append(sub_ext_formula)

        for i in range(count_param):
            for j in range(i):
                uni_formula = uni_formula & (
                    ~sub_aux_atoms[i] | ~sub_aux_atoms[j])
        or_sub_aux_atoms = QFFormula(False)
        for atom in sub_aux_atoms:
            or_sub_aux_atoms = or_sub_aux_atoms | atom
        uni_formula = uni_formula & or_sub_aux_atoms.equivalent(aux_atom)
        cardinality_constraint = (aux_pred, '=', len(self.domain) * count_param)
        return uni_formula, ext_formulas, cardinality_constraint, repeat_factor

    def _transform_counting_quantifier(self, formula: QuantifiedFormula) -> None:
        inner_formula = formula.quantified_formula

        if not isinstance(inner_formula, QuantifiedFormula):
            if not (isinstance(inner_formula, AtomicFormula) and inner_formula.pred.arity == 1):
                raise TypeError(
                    f"Unary counting quantifier requires a unary atomic formula inside, but got {inner_formula}"
                )

            quantifier_scope = formula.quantifier_scope
            comparator = quantifier_scope.comparator
            if comparator == 'mod':
                raise ValueError(
                    "Modulo unary counting quantifiers are only supported by "
                    "IncrementalWFOMC3Context"
                )
            count_param = int(quantifier_scope.count_param)
            self.cardinality_constraint.add_simple_constraint(
                inner_formula.pred, comparator, count_param
            )
            return

        logger.info(
            f"Handling binary counting formula with NEW encoding: {formula}"
        )
        comparator = formula.quantified_formula.quantifier_scope.comparator
        if comparator != '=':
            raise ValueError(
                f"Binary counting comparator '{comparator}' is not "
                "supported by WFOMCContext; use IncrementalWFOMC3Context "
                "for supported non-equality binary counting quantifiers"
            )

        uni_formula_old, ext_formulas_from_cnt, card_constraint, repeat_factor = \
            self._transform_forall_existsK(formula)
        k = int(formula.quantified_formula.quantifier_scope.count_param)

        skolem_preds = []
        skolem_axioms = top
        for ext_formula in ext_formulas_from_cnt:
            inner_formula = ext_formula.quantified_formula.quantified_formula
            skolem_pred = new_predicate(1, SKOLEM_PRED_NAME)
            skolem_preds.append(skolem_pred)
            axiom_body = skolem_pred(X) | ~inner_formula
            skolem_axioms &= axiom_body
            self.weights[skolem_pred] = (
                Rational(1, 1),
                Rational(-1, 1),
            )

        c_preds = [new_predicate(1, f"C_{j}_") for j in range(k + 1)]

        gamma_c_body = top
        for j in range(k + 1):
            true_atoms = [skolem_preds[h](X) for h in range(j)]
            false_atoms = [~skolem_preds[h](X) for h in range(j, k)]
            conj_true_A = reduce(lambda f1, f2: f1 & f2, true_atoms, top)
            conj_false_A = reduce(lambda f1, f2: f1 & f2, false_atoms, top)
            definition = conj_true_A & conj_false_A
            gamma_c_body &= c_preds[j](X).equivalent(definition)

        disjuncts = [p(X) for p in c_preds]
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
            self.weights[c_preds[j]] = (
                Rational(comb(k, j), 1),
                Rational(1, 1),
            )

        self.cardinality_constraint.add_simple_constraint(*card_constraint)
        self.repeat_factor *= repeat_factor

    def _apply_unary_evidence(self) -> None:
        plan = UnaryEvidencePlan.build(
            self.unary_evidence,
            self.domain,
            self.sentence,
            self.unary_evidence_strategy,
        )
        self.unary_evidence_plan = plan
        self.formula &= plan.formula

        if plan.cardinality_constraints:
            logger.info('Use cardinality constraints to encode unary evidence')
            logger.info('formula to encode unary evidence: {}', plan.formula)
            logger.info(
                'cardinality constraints: {}',
                plan.cardinality_constraints,
            )
            if not self.contain_cardinality_constraint():
                self.cardinality_constraint = CardinalityConstraint()
            self.cardinality_constraint.extend_simple_constraints(
                plan.cardinality_constraints
            )
        else:
            logger.info(
                "Prepared lifted unary evidence with {} evidence profile(s)",
                len(plan.partition.evidence_profiles),
            )
        self.repeat_factor *= plan.repeat_factor

    def _handle_linear_order_axiom(self) -> None:
        if self.problem.contain_linear_order_axiom():
            self.leq_pred = Pred('LEQ', 2)

        if self.problem.contain_predecessor_axiom():
            for pred in self.sentence.preds():
                if pred.name.startswith('PRED'):
                    if self.predecessor_preds is None:
                        self.predecessor_preds = {}
                    self.predecessor_preds[int(pred.name[4:])] = pred

        if self.problem.contain_circular_predecessor_axiom():
            if self.predecessor_preds is None:
                self.predecessor_preds = {
                    1: Pred('CIRCULAR_PRED', 2)
                }
            self.circular_predecessor_pred = Pred('CIRCULAR_PRED', 2)

    def _convert_weights_to_ring_elements(self) -> None:
        weights = []
        preds = []
        for pred, (w_true, w_false) in self.weights.items():
            preds.append(pred)
            weights.append(w_true)
            weights.append(w_false)

        ring_elements = to_ringelements(weights)
        for i, pred in enumerate(preds):
            self._weights[pred] = (
                ring_elements[2 * i],
                ring_elements[2 * i + 1],
            )

    def _build(self) -> None:
        self.sentence = tseitin_transform(self.sentence)
        logger.info('sentence after tseitin transform: \n{}', self.sentence)

        self.formula = self.sentence.uni_formula
        while not isinstance(self.formula, QFFormula):
            self.formula = self.formula.quantified_formula

        if self.unary_evidence:
            self._apply_unary_evidence()

        if self.sentence.contain_counting_quantifier():
            logger.info("Translating SC2 to SNF using the NEW encoding logic.")
            if not self.contain_cardinality_constraint():
                self.cardinality_constraint = CardinalityConstraint()
            for cnt_formula in self.sentence.cnt_formulas:
                self._transform_counting_quantifier(cnt_formula)

        self._skolemize()

        self._prepare_cardinality_weights()

        self._handle_linear_order_axiom()
        self._convert_weights_to_ring_elements()
