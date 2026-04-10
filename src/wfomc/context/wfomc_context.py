import math
from loguru import logger
from copy import deepcopy
from wfomc.fol.sc2 import SC2
from wfomc.fol.utils import new_predicate
from functools import reduce
from math import comb
from wfomc.network import CardinalityConstraint, PartitionConstraint, UnaryEvidenceEncoding, organize_evidence, \
    unary_evidence_to_ccs, unary_evidence_to_pc
from wfomc.fol.syntax import *
from wfomc.problems import WFOMCProblem
from wfomc.utils import Rational, RingElement


class WFOMCContext(object):
    """
    Context for WFOMC algorithm
    """

    def __init__(self, problem: WFOMCProblem,
                 unary_evidence_encoding: UnaryEvidenceEncoding = UnaryEvidenceEncoding.CCS):
        self.problem = deepcopy(problem)
        self.domain: set[Const] = self.problem.domain
        self.sentence: SC2 = self.problem.sentence
        self.weights: dict[Pred, tuple[RingElement,
                                       RingElement]] = self.problem.weights
        self.cardinality_constraint: CardinalityConstraint = self.problem.cardinality_constraint
        self.repeat_factor = 1
        self.unary_evidence = self.problem.unary_evidence
        self.unary_evidence_encoding = unary_evidence_encoding

        logger.info('sentence: \n{}', self.sentence)
        logger.info('domain: \n{}', self.domain)
        logger.info('weights:')
        for pred, w in self.weights.items():
            logger.info('{}: {}', pred, w)
        logger.info('cardinality constraint: {}', self.cardinality_constraint)
        logger.info('unary evidence: {}', self.unary_evidence)

        self.formula: QFFormula
        # for handling linear order axiom
        self.leq_pred: Pred = None
        self.predecessor_preds: dict[int, Pred] = None
        self.circular_predecessor_pred: Pred = None
        # for handling unary evidence
        self.element2evidence: dict[Const, set[AtomicFormula]] = dict()
        self.partition_constraint: PartitionConstraint = None
        self._build()
        logger.info('Skolemized formula for WFOMC: \n{}', self.formula)
        logger.info('weights for WFOMC: \n{}', self.weights)
        logger.info('repeat factor: {}', self.repeat_factor)
        logger.info('partition constraint: {}', self.partition_constraint)

    def contain_cardinality_constraint(self) -> bool:
        return self.cardinality_constraint is not None and \
            not self.cardinality_constraint.empty()

    def contain_partition_constraint(self) -> bool:
        return self.partition_constraint is not None

    def contain_existential_quantifier(self) -> bool:
        return self.sentence.contain_existential_quantifier()

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return (default, default)

    def decode_result(self, res: RingElement) -> Rational:
        if self.leq_pred is not None:
            res = res * Rational(math.factorial(len(self.domain)), 1)
        res = res / self.repeat_factor
        if self.contain_cardinality_constraint():
            res = self.cardinality_constraint.decode_poly(res)
        return res

    def _skolemize_one_formula(self, formula: QuantifiedFormula) -> QFFormula:
        """
        Only need to deal with \forall X \exists Y: f(X,Y) or \exists X: f(X,Y)
        """
        quantified_formula = formula.quantified_formula
        quantifier_num = 1
        while (not isinstance(quantified_formula, QFFormula)):
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
        while (not isinstance(formula, QFFormula)):
            formula = formula.quantified_formula
        for ext_formula in self.sentence.ext_formulas:
            formula = formula & self._skolemize_one_formula(ext_formula)
        return formula

    def _transform_forall_existsK(self, formula: QuantifiedFormula) -> \
        tuple[QFFormula, list[QuantifiedFormula], tuple, int]:
        uni_formula = top
        ext_formulas = []

        cnt_quantified_formula = formula.quantified_formula.quantified_formula
        cnt_quantifier = formula.quantified_formula.quantifier_scope
        count_param = cnt_quantifier.count_param

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

    def _transform_counting_quantifier(self, formula: QuantifiedFormula):
        """
        This function is responsible for implementing Lemma 4 from the Beat paper, which decomposes ∀X ∃=k Y: R(X,Y). During the decomposition process, it generates k independent existential quantifier formulas ∀X ∃Y: fᵢ(X,Y) that need to be satisfied.
        """

        inner_formula = formula.quantified_formula

        if not isinstance(inner_formula, QuantifiedFormula):
            if not (isinstance(inner_formula, AtomicFormula) and inner_formula.pred.arity == 1):
                raise TypeError(
                    f"Unary counting quantifier requires a unary atomic formula inside, but got {inner_formula}")

            quantifier_scope = formula.quantifier_scope
            comparator = quantifier_scope.comparator
            count_param = int(quantifier_scope.count_param)

            predicate_to_constrain = inner_formula.pred
            cardinality_constraint = (
                predicate_to_constrain, comparator, count_param)
            self.cardinality_constraint.add_simple_constraint(*cardinality_constraint)
            self.repeat_factor *= math.factorial(count_param)
        else:
            logger.info(
                f"Handling binary counting formula with NEW encoding: {formula}"
            )
            # 2.3.1 Call the old conversion function to get all components of the old encoding Γ*
            (
                uni_formula_old,
                ext_formulas_from_cnt,
                card_constraint,
                repeat_factor,
            ) = self._transform_forall_existsK(formula)
            # uni_formula_old: formula containing fᵢ decomposition and mutual exclusion constraints
            # ext_formulas: k formulas of the form ∀X∃Y:fᵢ(X,Y), which are the "ingredients" for Aᵢ predicates
            # card_constraint: cardinality constraint, e.g., |R|=n*k
            # repeat_factor: correction factor for the old encoding (k!)^n

            k = formula.quantified_formula.quantifier_scope.count_param  # Extract the counting parameter k and pre-create predicates needed for the new encoding

            # 2.3.2 Immediately handle the k existential quantifier formulas related to this counting quantifier
            skolem_preds = []  # Used to store the k Aᵢ predicates
            skolem_axioms = top  # Used to collect the k Skolemization axioms
            for (
                ext_formula
            ) in ext_formulas_from_cnt:  # Iterate over the k formulas of the form ∀X∃Y:fᵢ(X,Y)
                inner_formula = (
                    ext_formula.quantified_formula.quantified_formula
                )  # Extract the fᵢ(X,Y) part
                skolem_pred = new_predicate(
                    1, SKOLEM_PRED_NAME
                )  # Create a unique Skolem predicate (Aᵢ) for the current fᵢ
                skolem_preds.append(skolem_pred)  # Save Aᵢ for later use
                X = Var("X")
                axiom_body = (
                    skolem_pred(X) | ~inner_formula
                )  # Generate Skolemization axiom: Aᵢ(X) ∨ ¬fᵢ(X,Y)
                skolem_axioms &= axiom_body  # Add the axiom to the collection
                self.weights[skolem_pred] = (
                    Rational(1, 1),
                    Rational(-1, 1),
                )  # Set the weight (1, -1) for the newly created Aᵢ
            # 2.3.3 【Introducing Cᵢ】Create k + 1 new "standard state" marking predicates Cᵢ
            c_preds = [new_predicate(1, f"C_{j}_")
                       for j in range(k + 1)]

            # 2.3.4 【Establishing Γ꜀ Constraints】Precisely defining the equivalence relationship between each Cⱼ and Aᵢ combination
            gamma_c_body = top
            X = Var("X")
            for j in range(k + 1):
                # Construct the right side of the equivalence (⇔): the first j Aᵢ are true, the remaining k-j Aᵢ are false
                # Use reduce and & operators to build the conjunction chain for cleaner code
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
                )  # Add the rule Cⱼ(X) ⇔ definition to the overall constraint

            # 2.3.5 【Constructing the Forced Disjunction】Construct the ∀x ⋁ Cⱼ(x) rule, forcing the model to be standard
            disjuncts = [
                p(X) for p in c_preds
            ]  # Use the `reduce` function and the `|` operator to construct the disjunction. This part implements the formula in the paper: ∀x ⋁ Cⱼ(x)
            final_disjunction_body = reduce(
                lambda f1, f2: f1 | f2, disjuncts, bot
            )

            # 2.3.6 【Combination】 Combine all the parts into the final quantifier-free formula
            self.formula &= (
                uni_formula_old
                & skolem_axioms
                & gamma_c_body
                & final_disjunction_body
            )

            # 2.3.7 【Weighting】Set special "magic weights" for Cᵢ: w(Cⱼ) = C(k, j)
            for j in range(k + 1):
                weight = Rational(comb(k, j), 1)
                neg_weight = Rational(1, 1)
                self.weights[c_preds[j]] = (weight, neg_weight)

            # 2.3.8 [Update] Update the global cardinality constraints and correction factors
            self.cardinality_constraint.add_simple_constraint(
                *card_constraint)
            self.repeat_factor *= repeat_factor

    def _build(self):
        """
        Rewrite the _build method to handle different quantifiers separately.
        """
        # Step 1: Initialize and get the quantifier-free part of the formula (same as old method)
        self.formula = self.sentence.uni_formula
        while not isinstance(self.formula, QFFormula):
            self.formula = self.formula.quantified_formula

        if self.unary_evidence:
            self.element2evidence = organize_evidence(self.unary_evidence)
            if self.unary_evidence_encoding == UnaryEvidenceEncoding.PC:
                logger.info('Use partition constraint to encode unary evidence')
                evi_formula, partition = unary_evidence_to_pc(
                    self.element2evidence, self.domain
                )
                logger.info('formula to encode unary evidence: {}', evi_formula)
                logger.info('partition constraint: {}', partition)
                self.formula = self.formula & evi_formula
                self.partition_constraint = partition
            elif self.unary_evidence_encoding == UnaryEvidenceEncoding.CCS:
                logger.info('Use cardinality constraint to encode unary evidence')
                evi_formula, ccs, repeat_factor = unary_evidence_to_ccs(
                    self.element2evidence, self.domain
                )
                logger.info('formula to encode unary evidence: {}', evi_formula)
                logger.info('cardinality constraints: {}', ccs)
                self.formula = self.formula & evi_formula
                if not self.contain_cardinality_constraint():
                    self.cardinality_constraint = CardinalityConstraint()
                self.cardinality_constraint.extend_simple_constraints(ccs)
                self.repeat_factor *= repeat_factor

        # Step 2: Use the new encoding to handle counting quantifiers (core modification)
        if self.sentence.contain_counting_quantifier():
            logger.info("Translating SC2 to SNF using the NEW encoding logic.")
            if not self.contain_cardinality_constraint():
                self.cardinality_constraint = CardinalityConstraint()
            for cnt_formula in self.sentence.cnt_formulas:  # Iterate over all counting quantifier formulas in the sentence
                self._transform_counting_quantifier(cnt_formula)

        if self.contain_cardinality_constraint():
            self.cardinality_constraint.build()

        # Only process original existential quantifier formulas unrelated to cardinality quantifiers
        for ext_formula in self.sentence.ext_formulas:
            self.formula &= self._skolemize_one_formula(ext_formula)

        # self.formula = self.formula.simplify()

        if self.contain_cardinality_constraint():
            self.weights.update(
                self.cardinality_constraint.transform_weighting(
                    self.get_weight,
                )
            )

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
