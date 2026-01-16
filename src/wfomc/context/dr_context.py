from __future__ import annotations
from copy import deepcopy
import hashlib
from collections import defaultdict
import math
import numpy as np
from logzero import logger
from wfomc.fol.sc2 import SC2
from wfomc.fol.utils import new_predicate, convert_counting_formula
from wfomc.fol.syntax import (
    AUXILIARY_PRED_NAME,
    X,
    Y,
    AtomicFormula,
    Const,
    Pred,
    QFFormula,
    top,
    a,
    b,
    Top,
)
from wfomc.network.constraint import CardinalityConstraint
from wfomc.fol.syntax import *
from wfomc.problems import WFOMCProblem
from wfomc.utils.polynomial_flint import RingElement, Rational
from itertools import product


class DRWFOMCContext(object):
    def __init__(self, problem: WFOMCProblem): # problem (WFOMCProblem): Instance of WFOMC problem
        # Domain, sentence, weights, and cardinality constraints
        self.problem: WFOMCProblem = deepcopy(
            problem)  # Deep copy the input WFOMC problem to avoid modifying the original object.
        self.domain: set[Const] = problem.domain  # Domain
        self.sentence: SC2 = problem.sentence  # Logical sentence: extract the logical sentence in SC2 format.
        self.weights: dict[Pred, tuple[Rational, Rational]
                           ] = problem.weights  # Weights
        self.cardinality_constraint: CardinalityConstraint = (
            problem.cardinality_constraint
        )  # Cardinality constraints
        self.repeat_factor: int = 1  # Repeat factor, default is 1: Initialize a repeat factor for counting adjustments in special cases.

        logger.info("sentence: \n%s", self.sentence)
        logger.info("domain: \n%s", self.domain)
        logger.info("weights:")
        for pred, w in self.weights.items():
            logger.info("%s: %s", pred, w)
        logger.info("cardinality constraint: %s", self.cardinality_constraint)

        self.formula: QFFormula  # Quantifier-free formula: declare an instance variable to store the processed quantifier-free formula.
        # --- Handle linear order axiom
        if problem.contain_linear_order_axiom():  # Check if the problem contains a linear order axiom.
            self.leq_pred: Pred = Pred("LEQ", 2)  # If so, define a binary predicate named "LEQ" to represent it.
        else:
            self.leq_pred: Pred = None
        self.uni_formula: QuantifiedFormula = Top # Universal formula: initialize a variable to store the pure universal part of the formula, initially "Top" (true).
        self.ext_preds: list[QuantifiedFormula] = []  # List of existentially quantified predicates
        # --- cc Handle counting quantifiers, divided into single and double layers, each layer has mod = <=. Single layer means the predicate is unary, double layer means the predicate is binary
        self.cnt_preds: list[QuantifiedFormula] = [] # Counting predicate list: initialize a list to store auxiliary predicates introduced for handling counting quantifiers.
        self.cnt_params: list[int] = [] # Counting parameter k (int): initialize a list to store the parameters of counting quantifiers (e.g., k in ∃=k).
        self.cnt_remainder: list[int] = [] # Remainder r (int): initialize a list to store the remainders of modular counting quantifiers (e.g., r in ∃≡r (mod k)).
        # --- unary variables related to unary constraints
        self.mod_pred_index: list[int] = []  # Index of modular predicates
        self.exist_mod: bool = False  # Whether modular arithmetic exists
        self.unary_mod_constraints: list[tuple] = []  # Unary modular constraints [(Pred, r, k), …]
        self.unary_eq_constraints: list[tuple] = []  # [(pred, k), ...]
        self.unary_le_constraints: list[tuple] = []  # [(pred, k_max), ...]
        # --- Handle <=
        self.exist_le: bool = False  # Whether there is <=
        self.le_pred: list[Pred] = []  # List of less than or equal predicates
        self.le_index: list[int] = []  # Index of less than or equal predicates
        # Comparator handler mapping
        self.comparator_handlers: dict[str, callable] = {  # Comparator handler mapping
            "mod": self._handle_mod,
            "=": self._handle_eq,
            "<=": self._handle_le,
        }
        self._build()  # Preprocess logical formula: call the _build method to start converting and decomposing the logical formula.
        self.c_type_shape: tuple = tuple()
        self.build_c_type_shape()
        self.binary_evidence: list = []
        self.build_binary_evidence()
        # binary cardinality_constraints is underconstruction .This part employs symbolic weights.
        self.card_preds: list = []
        self.card_ccs: list = []
        self.card_vars: list = []
        self.build_cardinality_constraints()
        self.build_repeat_factor() # Update repeat factor, for example, the input example m-odd needs to be divided by domain size, which is n choose 1.

    def build_repeat_factor(self):
        """
        For the example of the odd-degree input, it needs to be divided by the domain size.
        Check whether the one-variable equality constraints of "odd" and "U" appear in the constraints. If both exist, set the repeat_factor to the domain size.
        """
        if hasattr(self, 'unary_eq_constraints') and self.unary_eq_constraints: 
            constraint_names = {
                constraint[0].name for constraint in self.unary_eq_constraints}
            if 'Odd' in constraint_names and 'U' in constraint_names:
                self.repeat_factor = len(self.problem.domain)
                print("change repeat factor to:", self.repeat_factor)

    def stop_condition(self, target_c):
        """
            Check whether the final state of the target element satisfies all constraints.
        Args:
            target_c: target element's c type

        Returns:
            bool: 
        """
        pred_state = target_c[1:] # Obtain the current c type of the element, which is also the number of items that still need to be connected
        if self.exist_le: # Check whether the problem contains <=k type counting quantifiers, as their handling logic is different.
            for i in range(len(pred_state)): 
                if i not in self.le_index and pred_state[i] != 0: # If the current index i does not belong to the relaxed <=k constraint, and the counting state is not zero
                    return False # Then the strict constraint is not satisfied, immediately judge failure.
        else: # --- No <=k constraints ---
            return all(i == 0 for i in pred_state) # Check whether the target's corresponding c type does not need to be connected

    def _extract_formula(self, formula):
        """
        Extract the type, core formula, and quantifier scope of the counting quantifier formula
        """
        # This line checks whether the quantified_formula attribute inside the passed formula is still an instance of the QuantifiedFormula class. If it is, it indicates that this is a nested quantified formula (i.e., a "binary" structure).
        if isinstance(formula.quantified_formula, QuantifiedFormula):
            inner_formula = formula.quantified_formula  # Assign the internal quantified formula to the inner_formula variable.
            return (
                "binary",  # "binary": string, indicating this is a binary/nested type formula.
                inner_formula.quantified_formula,  # Extract the "core" formula without quantifiers.
                inner_formula.quantifier_scope,  # Extract the scope information of the quantifier.
            )
        else:
            return (
                "unary",  # "unary": string, indicating this is a unary/single-layer type formula.
                formula.quantified_formula,  # Extract the "core" formula without quantifiers.
                formula.quantifier_scope,  # Extract the scope information of the quantifier.
            )

    def _add_aux_equiv(self, inner_formula):
        aux_pred = new_predicate(2, AUXILIARY_PRED_NAME) # Create a new binary auxiliary predicate
        self.uni_formula = self.uni_formula & inner_formula.equivalent(
            aux_pred(X, Y)
        ) # Add the equivalence relation to the main formula
        self.cnt_preds.append(aux_pred) # Add the auxiliary predicate to the counting predicate list
        return aux_pred

    def _handle_mod(self, type, idx, inner_formula, qscope, param, _):
        """
        Handling the existential quantifier ∃_{≡r (mod k)} Args:
            type: unary or binary
            idx: index
            inner_formula: internal formula
            qscope: quantifier scope
            param: parameter (r, k)
            comparator: comparator
        """
        r, k = param
        # unary is under construction
        if (
            type == "unary"
            and isinstance(inner_formula, AtomicFormula)
            and inner_formula.pred.arity == 1
            and inner_formula.args == (qscope.quantified_var,)
        ):
            self.unary_mod_constraints.append(
                (inner_formula.pred, r, k)
            )
            return
        elif type == "binary":
            # binary mod
            self.exist_mod = True # Existential quantification operation
            self.mod_pred_index.append(idx) # Record the index of the mod predicate
            self.cnt_remainder.append(r) # Record the remainder r
            self.cnt_params.append(k) # Record the parameter k
            self._add_aux_equiv(inner_formula) # Add auxiliary equivalence predicate
    def _handle_eq(self, type, idx, inner_formula, qscope, param, comparator):
        """
        Handling the existential quantifier ∃_{=m}
        Args:
            type: unary or binary
            idx: index
            inner_formula: internal formula
            qscope: quantifier scope
            param: parameter k
            comparator: comparator
        """
        # unary ∃_{=k} X  A(X)
        if (
            type == "unary"
            and isinstance(inner_formula, AtomicFormula)
            and inner_formula.pred.arity == 1
            and inner_formula.args == (qscope.quantified_var,)
        ): # Unary equality constraint
            self.unary_eq_constraints.append(
                (inner_formula.pred, param)
            ) # Record unary equality constraint
            return # Note: These predicates do not need to be added to the cnt_preds list, as they are handled as separate constraints in the algorithm.
        elif type == "binary":
            self.cnt_remainder.append(None) # Binary equality has no remainder
            self.cnt_params.append(param) # Record parameter k
            self._add_aux_equiv(inner_formula) # Add auxiliary equivalence predicate
    def _handle_le(self, type, idx, inner_formula, qscope, param, comparator):
        if (
            type == "unary"
            and isinstance(inner_formula, AtomicFormula)
            and inner_formula.pred.arity == 1
            and inner_formula.args == (qscope.quantified_var,)
        ): # under construction
            self.unary_le_constraints.append(
                (inner_formula.pred, param)
            )
            return
        elif type == "binary": # binary <=
            self.cnt_remainder.append(None) # Binary <= has no remainder
            self.cnt_params.append(param) # Record parameter k
            aux_pred = self._add_aux_equiv(inner_formula) # Add auxiliary equivalence predicate
            self.le_pred.append(aux_pred) # Record less than or equal predicate
            self.exist_le = True # Mark existence of <= constraint
    def _build(self):
        """
        Build and preprocess the core functions. 
        This method is responsible for decomposing and converting the original SC2 logical sentences into an internal representation that can be processed by the algorithm. It mainly performs the following operations:
            1. Extract the pure universal part (without quantifier normal form) from the formula.
            2. Skolemize existential quantifiers (ext_formulas), introduce new Skolem predicates, and add constraints to the main formula.
            3. Classify and process quantifier formulas (cnt_formulas):
                - Unary quantifiers are extracted as independent constraints (such as unary_eq_constraints).
                - Binary quantifiers are handled by introducing auxiliary predicates and adding equivalence relations to the main formula.
            4. Integrate all the newly generated auxiliary predicates (Skolem predicates and counting predicates), and establish an index for them.
        """
        # 1. Extract pure universal quantifier-free formula (QF-Form)
        # Repeatedly strip away the outermost universal quantifiers until only the core non-quantifier formula remains.
        self.uni_formula = self.sentence.uni_formula
        while not isinstance(self.uni_formula, QFFormula):
            self.uni_formula = self.uni_formula.quantified_formula

        # Extract the quantifier and count noun parts from the sentence.
        ext_formulas = self.sentence.ext_formulas  # Existential quantifier formulas
        cnt_formulas = self.sentence.cnt_formulas  # Counting quantifier formulas

        # 2. Handle existential quantifiers (Skolemization)
        # Iterate over all existential quantifier formulas, eliminate them through Skolemization, and merge the generated Skolem constraints (&) into the main formula.
        for formula in ext_formulas:
            self.uni_formula = self.uni_formula & self._skolemize_one_formula(
                formula)

        # 3. Handle counting quantifiers
        # Iterate over all counting quantifier formulas, dispatching them based on their type (unary/binary) and comparator (=, <=, mod).
        for idx, formula in enumerate(cnt_formulas):
            # Parse the structure of the counting quantifier: type (unary/binary), core formula, quantifier scope
            type, inner_formula, qscope = self._extract_formula(
                formula
            )  # Return the single or double-layer type "type", the kernel formula "inner_formula", and the quantifier scope "qscope"
            comparator = qscope.comparator  # Get comparator 'mod' / '=' / '<=' / ...
            cnt_param_raw = qscope.count_param  # Get counting parameter (r,k) or int

            # Use the dispatch table self.comparator_handlers to call the corresponding handler function (e.g., _handle_eq).
            # Note: idx uses the current length of self.cnt_preds. This is because unary constraints do not add new counting predicates to the cnt_preds list, ensuring that new binary counting predicates are assigned continuous and correct indices.
            idx = len(self.cnt_preds)  # Use the current length of cnt_preds as the index
            # Note that the lengths of cnt_formulas and cnt_preds differ. Unary mod does not add to cnt_preds. To skip unary mod, do not manually increment idx = idx + 1. This is because idx must always stay synchronized with the current length of cnt_preds, keeping new predicate indices continuous and correct,
            self.comparator_handlers[comparator](
                type, idx, inner_formula, qscope, cnt_param_raw, comparator
            )

        # 4. Integrate all auxiliary predicates and build indices
        # Merge new predicates generated by Skolemization (ext_preds) and those generated by handling binary counting quantifiers (cnt_preds).
        self.all_preds = self.ext_preds + self.cnt_preds
        self._pred2idx = {
            pred: i for i, pred in enumerate(self.all_preds)
        }  # Then create a mapping dictionary _pred2idx from predicate objects to their indices in the total list for quick lookup later.
        self.le_index = [
            self.all_preds.index(pred) for pred in self.le_pred
        ]  # If the formula contains counting quantifiers of the type <=k, their corresponding auxiliary predicates are recorded in self.le_pred. This line of code finds the indices of these special predicates in the total list self.all_preds and saves them, as the algorithm has special handling logic for them.

    def _skolemize_one_formula(self, formula: QuantifiedFormula) -> QFFormula:
        """Skolemize a single existential quantifier formula.

        Skolemization is a standard logical technique to eliminate existential quantifiers. This function transforms a formula of the form ∃Y: φ(X,Y)
        into a logically equivalent (in terms of satisfiability) formula without existential quantifiers.
        It achieves this by introducing a new Skolem predicate S and generating constraints such as φ(X,Y) → S(X).

        Main steps:
        1.  Parse the input formula to determine the nesting depth of existential quantifiers (quantifier_num).
        2.  If the kernel formula φ itself is complex, introduce an auxiliary predicate @aux to simplify it and add an equivalence constraint (φ ↔ @aux).
        3.  Create a new Skolem predicate S based on the external variables (such as X here).
        4.  Add the core Skolem constraint of the form S ∨ ¬φ (equivalent to φ → S).
        5.  Assign a special weight (1, -1) to the new Skolem predicate S, which is a technique to enforce logical constraints through weighting.

        Args:
            formula (QuantifiedFormula): An undetermined existential quantifier formula.

        Returns:
            QFFormula: A constraint formula without existential quantifiers generated after Skolemization.
        """
        quantified_formula = formula.quantified_formula  # Obtain the internal formula within the quantifier, for example, for ∃Y: P(X,Y), the obtained result is P(X,Y).
        quantifier_num = 1  # Initialize the number of quantifiers to 1. This variable is used to record the nesting depth of existential quantifiers to determine the arity of the Skolem predicate.
        while not isinstance(
            quantified_formula, QFFormula
        ):  # This loop handles nested existential quantifiers, for example, ∃Y ∃Z: P(X,Y,Z).
            quantified_formula = quantified_formula.quantified_formula
            quantifier_num += 1  # It continues to strip off quantifiers until it finds the innermost quantifier-free formula (QFFormula) and counts the number of layers stripped.

        skolem_formula: QFFormula = top  # Initialize an empty Skolem formula, initially set to "Top" (logical true).
        # ext_formula refers to the quantifier-free core formula inside the existential quantifier, for example, P(X,Y)
        ext_formula = quantified_formula
        if not isinstance(
            ext_formula, AtomicFormula
        ):  # If the core formula is not a simple atomic formula (e.g., it is a compound formula A(X) & B(Y)), an auxiliary predicate Z is introduced to simplify it.
            aux_pred = new_predicate(
                quantifier_num, AUXILIARY_PRED_NAME
            )  # Create a new auxiliary predicate whose arity is determined by the number of quantifiers.
            aux_atom = (
                aux_pred(X, Y) if quantifier_num == 2 else aux_pred(X)
            )  # Create an atomic formula based on the arity, e.g., aux(X,Y) or aux(X).
            skolem_formula = skolem_formula & (
                ext_formula.equivalent(aux_atom)
            )  # Add the equivalence constraint "core formula <=> auxiliary atom" to the Skolem formula.
            ext_formula = (
                aux_atom
            )  # Subsequent processing will directly use this simpler auxiliary atomic formula.

        # Create Skolem predicates and corresponding atoms based on the number of quantifiers.
        # The arity here depends on the number of external universal quantifier variables; the code assumes at most one (i.e., X).
        if quantifier_num == 2:  # Corresponds to ∀X ∃Y ...
            skolem_pred = new_predicate(
                1, SKOLEM_PRED_NAME
            )  # Create a unary Skolem predicate S(X).
            skolem_atom = skolem_pred(X)
        elif quantifier_num == 1:  # Corresponds to ∃Y ... (no external universal quantifier)
            skolem_pred = new_predicate(
                0, SKOLEM_PRED_NAME
            )  # Create a zero-arity Skolem predicate S(), i.e., a proposition.
            skolem_atom = skolem_pred()

        skolem_formula = skolem_formula & (
            skolem_atom | ~ext_formula
        )  # This is equivalent to P(X,Y) → S(X).
        self.weights[skolem_pred] = (
            Rational(1, 1),
            Rational(-1, 1),
        )  # Set weights for the newly created Skolem predicate.
        return skolem_formula

    def build_c_type_shape(self):
        """Construct the dimensional information of the counting state space c type."""
        self.c_type_shape = list(
            2 for _ in self.ext_preds
        )  # For each existential quantifier predicate, the state has two possibilities (true/false).
        for idx, k in enumerate(self.cnt_params):  # For each counting quantifier.
            if idx in self.mod_pred_index:  # If it is a modulus type.
                self.c_type_shape.append(k)  # The state space size is k (0 to k-1).
            else:  # If it is =k or <=k type.
                self.c_type_shape.append(k + 1)  # The state space size is k+1 (0 to k).

    def build_binary_evidence(self):
        """
        Generate all possible truth assignments for binary predicates between two abstract elements a and b.
        These combinations are called "binary evidence" and are used to calculate the weights of state transitions later.
        """
        ext_atoms = list(
            (
                (~pred(a, b), ~pred(b, a)),  # Combination 1: pred(a,b) is false, pred(b,a) is false
                (~pred(a, b), pred(b, a)),  # Combination 2: pred(a,b) is false, pred(b,a) is true
                (pred(a, b), ~pred(b, a)),  # Combination 3: pred(a,b) is true, pred(b,a) is false
                (pred(a, b), pred(b, a)),  # Combination 4: pred(a,b) is true, pred(b,a) is true
            )
            # Generate these 4 combinations for each binary predicate pred in self.ext_preds.
            for pred in self.ext_preds[::-1]
        )
        cnt_atoms = list(
            (
                (~pred(a, b), ~pred(b, a)),
                (~pred(a, b), pred(b, a)),
                (pred(a, b), ~pred(b, a)),
                (pred(a, b), pred(b, a)),
            )
            for pred in self.cnt_preds[::-1]
        ) # Perform the same operation for all auxiliary predicates introduced by binary counting quantifiers.
    
        # Use itertools.product to compute the Cartesian product of all predicate truth value combinations. This generates an iterator that produces a complete combination of truth values for all predicates each time. For example, if there are 2 predicates, each with 4 cases, this will produce 4*4=16 final combinations.
        for atoms in product(*cnt_atoms, *ext_atoms):
            self.binary_evidence.append(
                frozenset(sum(atoms, start=()))
            )

    def build_cardinality_constraints(self):
        if self.contain_cardinality_constraint():
            self.cardinality_constraint.build()
            self.weights.update(
                self.cardinality_constraint.transform_weighting(
                    self.get_weight,
                )
            )

    def decode_result(self, res: RingElement) -> Rational:
        """Decode the final result, applying cardinality constraints and repetition factor adjustments."""
        if not self.contain_cardinality_constraint(): # If there are no cardinality constraints, simply return the result divided by the repetition factor.
            res = res / self.repeat_factor
        else: # If there are cardinality constraints, first process the result through the symbolic weight decoding method of the cardinality constraint, then divide by the repetition factor.
            res = self.cardinality_constraint.decode_poly(
                res) / self.repeat_factor
        if self.leq_pred is not None: # If there is a linear order axiom, multiply by the factorial of the domain size.
            res *= Rational(math.factorial(len(self.domain)), 1)
        return res

    def contain_cardinality_constraint(self) -> bool:
        return (
            self.cardinality_constraint is not None
            and not self.cardinality_constraint.empty()
        )

    def contain_linear_order_axiom(self) -> bool:
        """Check if the problem contains a linear order axiom."""
        return self.problem.contain_linear_order_axiom()

    def build_t_update_dict(self, r, n_cells):
        """Build the state transition lookup table t_update_dict.
        Construct a large and detailed state transition lookup table.
        This table stores how any two elements (in any possible states) transition to new states when connected,
        and what the weight of this transition is.
        By precomputing all these possibilities, the core recursive algorithm does not need to dynamically compute state transitions at runtime,
        it only needs to quickly look up in this table, greatly improving performance.
        Args:
            r (dict): The relation dictionary computed in build_weight. It stores the possible state changes (dt, reverse_dt) and their weights when cell i and cell j interact.
            n_cells (int): The total number of cell types.
        Returns:
            defaultdict: A nested defaultdict with the structure:
                         t_update_dict[(c1, c2)][(c1_new, c2_new)] = weight
                         where (c1, c2) are the full states of the two elements before pairing,
                         and (c1_new, c2_new) are the new states after pairing.
        """
        # --- Initialize the lookup table
        t_update_dict = defaultdict(
            lambda: defaultdict(lambda: Rational(0, 1))
        ) # Create a nested defaultdict. Structure: t_update_dict[(c1, c2)][(c1_new, c2_new)] = weight. Where (c1, c2) are the states of the two elements before pairing, and (c1_new, c2_new) are the new states after pairing.

        # --- Precompute all possible c type state combinations, where t corresponds to c type
        if self.exist_mod:
            final_list = [
                tuple(range(2)) for _ in self.ext_preds
            ]
            for idx, k in enumerate(self.cnt_params):
                if idx in self.mod_pred_index:
                    final_list += [tuple(range(k))]
                else:
                    final_list += [tuple(range(k + 1))]
            all_ts = list(product(*(final_list))) # Use itertools.product to compute the Cartesian product of all state ranges in final_list. This generates all possible state vector combinations. For example, if final_list is [(0,1), (0,1,2)], product will generate (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).
        else:
            all_ts = list(
                product(
                    *(
                        [tuple(range(2)) for _ in self.ext_preds]
                        + [tuple(range(i + 1)) for i in self.cnt_params]
                    )
                )
            )
        #
        # Iterate over all possibilities and fill the lookup table
        for i in range(n_cells): # Iterate over the first element's cell type i.
            for j in range(n_cells): # Iterate over the second element's cell type j.
                for t1 in all_ts: # Iterate over all possible states t1 of the first element.
                    for t2 in all_ts: # Iterate over all possible states t2 of the second element.
                        for (dt, reverse_dt), rijt in r[(i, j)].items(): #  r is the relation dictionary computed in build_weight. It stores the possible state *changes* (dt, reverse_dt) and their weights rijt when cell i and cell j interact.
                            # --- Compute new states
                            t1_new = [a - b for a, b in zip(t1, dt)] # Compute new state by subtracting the change from the original state.
                            t2_new = [a - b for a, b in zip(t2, reverse_dt)]
                            # 
                            # --- Handle modulus constraints
                            if self.exist_mod:
                                for p, k_i in enumerate(self.cnt_params): # Iterate over all counting quantifiers
                                    index = (len(self.ext_preds) + p) # Get the index position of the counting quantifier in the c type
                                    if p in self.mod_pred_index: # If this is a modulus constraint
                                        t1_new[index] %= k_i 
                                        t2_new[index] %= k_i
                            # --- Pruning: Check if states are valid
                            if any(
                                t1_new[len(self.ext_preds) + p] < 0
                                or t2_new[len(self.ext_preds) + p] < 0
                                for p in range(len(self.cnt_params))
                            ): # Check if the states of ordinary counting constraints become negative.
                                continue  # If the state becomes negative, this is an invalid transition, skip it.
                            # --- Handle existential quantifiers
                            for idx in range(
                                len(self.ext_preds)
                            ):
                                t1_new[idx] = max(t1_new[idx], 0) # Keep them non-negative.
                                t2_new[idx] = max(t2_new[idx], 0)
                            # --- Assemble the complete c type
                            c1 = (i,) + t1
                            c2 = (j,) + t2
                            c1_new = (i,) + tuple(t1_new)
                            c2_new = (j,) + tuple(t2_new)
                            # --- Update the lookup table
                            t_update_dict[(c1, c2)][
                                (c1_new, c2_new)
                            ] += rijt
        return t_update_dict

    def build_weight(self, cells, cell_graph):
        """
        Construct the dictionary of weights and relationships r
        This function calculates the weight dictionary and the relationship dictionary based on the given cells and cell diagrams.
        Used for subsequent state transition calculations.
        Args:
            cells: List of cell types, a list containing all Cell types
            cell_graph: An object that precomputes the weights of Cells and Cell pairs
        Returns:
            tuple: (w2t dictionary, w weight dictionary, r relationship dictionary)
                   w2t: A dictionary mapping from cell index to predicate state dictionary, e.g., {0: (1, 0, 1, 2), 1: (0, 1, 0, 1)}, values represent the required quantities
                   w: A weight dictionary for each cell type, e.g., {0: Rational(1, 1), 1: Rational(2, 1)}, keys are cell indices, values are weights
                   r: A relationship dictionary between cell pairs,
        """

        n_cells = len(cells)  # Get the total number of Cell types.
        w2t = dict() # Initialize the w2t dictionary to store the mapping from cell index to its target state vector.
        w = defaultdict(
            lambda: Rational(0, 1)
        )  # Initialize the w dictionary to store the weight of each Cell.
        r = defaultdict(
            lambda: defaultdict(lambda: Rational(0, 1))
        )  # Initialize the r dictionary. This is a nested dictionary with the structure r[(i, j)][(t, reverse_t)] = weight, used to store the weight from Cell pair (i, j) to a specific state transition (t, reverse_t).
        for i in range(n_cells):  # Iterate over all cells, index i
            cell_weight = cell_graph.get_cell_weight(cells[i])  # Get the weight of the current cell
            logger.debug("Cell %d weight: %s", i, cell_weight)
            t = list()  # Initialize the state list t to store the predicate states (1 = true, 0 = false)

            # The target state of calculating existential quantifiers (Skolem predicates)
            for (
                pred
            ) in self.ext_preds:
                if cells[i].is_positive(pred): # Check if the Skolem predicate pred is true for elements of type i.
                    t.append(0) # True, indicating that the existential quantifier requirement is already satisfied and no additional relationship support is needed.
                else:
                    t.append(1)  
            logger.debug("Cell %d existential quantifier state: %s", i, t)
            #
            # Counting quantifiers
            # Logical explanation: Suppose there is a constraint ∀X ∃_{=k} Y: B(X,Y), which is transformed into ∀X ∃_{=k} Y: @aux(X,Y). When considering an element d_i, we need to ensure that there are exactly k @aux relationships related to it. If cells[i] makes @aux(d_i, d_i) true (is_positive), then this element itself satisfies 1 relationship. Therefore, it needs to obtain k-1 relationships from the other N-1 elements. So the target state is set to param - 1. If @aux(d_i, d_i) is false, it needs to obtain all k relationships from the other N-1 elements. So the target state is set to param. The logic for mod constraints is similar, except that after subtraction, modulo is taken.
            for idx, (pred, param) in enumerate(zip(self.cnt_preds, self.cnt_params)): # Here the logic is to calculate how many relationships an element needs to obtain from other elements to satisfy the counting quantifier.
                if cells[i].is_positive(pred): # If it is true for itself, it has already satisfied 1 relationship.
                    if (
                        self.exist_mod and idx in self.mod_pred_index
                    ): # For mod constraints, the target becomes (r-1).
                        t.append(
                            self.cnt_remainder[idx] - 1
                        )
                    else: # For ordinary counting constraints (e.g., =k), the target becomes (k-1).
                        t.append(param - 1)
                else: # If it is false for itself, it needs to obtain all required relationships from other elements.
                    if (
                        self.exist_mod and idx in self.mod_pred_index
                    ): # For mod constraints, the target is r.
                        t.append(self.cnt_remainder[idx])
                    else: # For ordinary counting constraints, the target is k.
                        t.append(param)
            logger.debug("Cell %d counting quantifier state: %s", i, t)
            w2t[i] = tuple(t)  # Store the calculated target state vector t in the w2t dictionary.
            w[i] = w[i] + cell_weight # Accumulate the weight of this cell type into the w dictionary.
            #
            # Start calculating binary transition weights.
            for j in range(n_cells):  # Start an inner loop, iterate over all Cell types j, forming Cell pairs (i, j).
                cell1 = cells[i]
                cell2 = cells[j]
                for evi_idx, evidence in enumerate(
                    self.binary_evidence
                ): # Iterate over all possible "binary evidence" previously generated in build_binary_evidence.
                    # Initialize two lists, t and reverse_t, to represent the state changes caused by this interaction.
                    t = list()
                    reverse_t = (
                        list()
                    )
                    # Get the interaction weight between cell1 and cell2 under the given evidence from cell_graph.
                    two_table_weight = cell_graph.get_two_table_weight(
                        (cell1, cell2), evidence
                    )
                    # If the weight is 0, it means this interaction is impossible, so skip it.
                    if two_table_weight == Rational(
                        0, 1
                    ):
                        continue
                    
                    # Construct state change vectors t and reverse_t based on the evidence (a truth value combination).
                    # The bitwise operations here are an efficient decoding method to extract the truth value of each predicate from evi_idx.
                    for pred_idx, pred in enumerate(
                        self.ext_preds + self.cnt_preds
                    ):
                        # Check the truth value of pred(b,a) in the evidence.
                        if (
                            evi_idx >> (2 * pred_idx)
                        ) & 1 == 1:
                            reverse_t.append(1)
                        else:
                            reverse_t.append(0)
                        # Check the truth value of pred(a,b) in the evidence.
                        if (
                            evi_idx >> (2 * pred_idx + 1)
                        ) & 1 == 1:
                            t.append(1)
                        else:
                            t.append(0)
                    r[(i, j)][
                        (tuple(t), tuple(reverse_t))
                    ] = two_table_weight  # Store the relationship weight using the predicate state combination.
        return w2t, w, r  # Return the mapping dictionary, weight dictionary, and relationship dictionary

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return default, default

    def check_unary_constraints(self, config, mask) -> tuple[bool, bool, bool]:
        """
        Check if the given configuration violates unary constraints.
        Args:
            config (iterable): The state vector of the current configuration.
            mask (tuple): A triplet of unary constraint masks generated by build_unary_mask.
            
        Returns:
            tuple: Three boolean values indicating whether unary modular constraints, unary equality constraints, and unary less-than-or-equal constraints are violated, respectively.
        """
        return (
            self.check_unary_mod_constraints(config, mask[0]),
            self.check_unary_eq_constraints(config, mask[1]),
            self.check_unary_le_constraints(config, mask[2]),
        )

    def check_unary_mod_constraints(self, config, unary_mod_mask) -> bool:
        for mask, r_mod, k_mod in unary_mod_mask:            
            # `np.fromiter(config, dtype=np.int32)`: Efficiently converts the `config` object (which may be a custom wrapper class) into a NumPy array for mathematical operations.
            # `mask @ ...`: The `@` operator is the matrix multiplication operator in NumPy. Here, it performs the "dot product" between the mask vector `mask` and the configuration vector `config`. Since `mask` has 1s at positions satisfying the predicate and 0s elsewhere, the result of this dot product is exactly the total number of elements in the current configuration that satisfy the constraint predicate.
            config_total_unary_constraint = mask @ np.fromiter(
                config, dtype=np.int32
            )
            if config_total_unary_constraint % k_mod != r_mod: # Check if the computed total satisfies the modular constraint.
                return True # If not equal, it means this constraint is violated. The function immediately returns True, indicating "violation found".
        return False

    def check_unary_eq_constraints(self, config, unary_eq_mask) -> bool:
        for mask, k_eq in unary_eq_mask:
            if (mask @ np.fromiter(config, dtype=np.int32)) != k_eq: # Check if the computed total equals the specified k_eq.
                return True # If not equal, it means this constraint is violated. The function immediately returns True, indicating "violation found".
        return False

    def check_unary_le_constraints(self, config, unary_le_masks) -> bool:
        vec = np.fromiter(config, dtype=np.int32)
        for mask, k_max in unary_le_masks: 
            if (mask @ vec) > k_max: # Check if the computed total is less than or equal to the specified k_max.
                return True # If greater, it means this constraint is violated. The function immediately returns True, indicating "violation found".
        return False

    def build_unary_mask(self, cells) -> tuple[list, list, list]:
        """"Construct a unary constraint mask.
        Return three types of unary constraint masks as a triplet:
        1. List of unary modulus constraint masks
        2. List of unary equality constraint masks
        3. List of unary less than or equal constraint masks"
        """
        return (
            self.build_unary_mod_mask(cells),
            self.build_unary_eq_mask(cells),
            self.build_unary_le_mask(cells),
        )

    def build_unary_le_mask(self, cells) -> list:
        n_cells = len(cells)
        masks = []
        for pred, k_max in self.unary_le_constraints:
            mask = np.fromiter(
                (1 if cell.is_positive(pred) else 0 for cell in cells),
                dtype=np.int8,
                count=n_cells,
            )
            masks.append((mask, k_max))
        return masks

    def build_unary_mod_mask(self, cells) -> list:
        """Construct a list of unary modulus constraint masks.
        For each unary modulus constraint (e.g., ∃_{≡r (mod k)}x: P(x)), generate a mask vector.
        This mask can quickly compute the total number of elements satisfying predicate P in any configuration.
    
        cells (list): A list of all cell types.

        Returns:
            list: A list of masks, each element is a tuple (mask, r, k) where
                  mask is a NumPy array, and r and k are parameters of the modulus constraint.
        """
        n_cells = len(cells) # Get the total number of cell types to determine the length of the mask vector.
        masks = []
        for pred, r, k in self.unary_mod_constraints: # Iterate over all unary modulus constraints parsed and stored in the _build method. Each constraint is a tuple (pred, r, k).
            mask = np.fromiter(
                (
                    # This is a generator expression that iterates over all cell types.
                    # For each cell, it checks whether it satisfies the predicate pred of the current constraint.
                    # If it satisfies (cell.is_positive(pred) returns True), it generates 1; otherwise, it generates 0.
                    1 if cell.is_positive(pred) else 0 for cell in cells
                ),
                dtype=np.int8,
                count=n_cells,
            )
            masks.append((mask, r, k))
        return masks

    def build_unary_eq_mask(self, cells) -> list:
        n_cells = len(cells)
        masks = []
        for pred, k_eq in self.unary_eq_constraints:
            mask = np.fromiter(
                (1 if cell.is_positive(pred) else 0 for cell in cells),
                dtype=np.int8,
                count=n_cells,
            )
            masks.append((mask, k_eq))
        return masks


class ConfigUpdater:
    """
    A memoized configuration updater for efficient state transitions.
    Cache_H is a nested two-level dictionary:
        First-level dictionary:
            Key: A tuple (target_c, other_c), representing the initial states of two elements to be paired.
            Value: The second-level dictionary.
        Second-level dictionary:
            Key: An integer j, representing the number of other_c elements paired with target_c.
            Value: An H dictionary, which is the complete result of pairing target_c with j other_c elements.
    Example:
    Cache_H = {
        # Cached pairing results for (Type A, Type B)
        ( (7, 1), (10, 0) ): {
            1: { ... H_for_j_1 ... },  # Result of pairing A with 1 B
            2: { ... H_for_j_2 ... },  # Result of pairing A with 2 B
            3: { ... H_for_j_3 ... }   # Result of pairing A with 3 B
        },
    }
           
    H is a dictionary with the following structure:
        Key: A tuple (target_c_new, H_config_new)
            target_c_new: The new state coordinates of the target element after pairing (a tuple).
            H_config_new: A HashableArrayWrapper object. The array inside it records what new states each of the j other_c elements became after pairing.
        Value: A Rational object, representing the cumulative weight of achieving this specific state combination.
        
    Example：
        H = {
        ( (7, 0, ), <HashableArrayWrapper for [[0,0],[3,0]]> ) : Rational(12, 1),
        }
    """
    def __init__(self, t_update_dict, c1_type_shape, Cache_H):
        self.t_update_dict = t_update_dict # A pre-calculated large lookup table stores the state transition rules and weights when any two elements are paired.
        self.c1_type_shape = c1_type_shape
        self.Cache_H = Cache_H # A cache dictionary used to store already computed configuration update results, avoiding redundant calculations and improving efficiency.

    def f(self, target_c, other_c, l): # Core method, calculates the result of pairing target_c with l other_c elements.
        # --- 1. Intelligent cache lookup ---
        if (target_c, other_c) in self.Cache_H: # Check if the pairing of target_c and other_c types has been computed before.
            H_sub_dict = self.Cache_H[
                (target_c, other_c)
            ] # If yes, get the sub-cache for this pairing type. This sub-cache stores results by the number of pairings l.
            num_start = l # Start from the required number l and look backward for the nearest cached result.
            while (
                num_start not in H_sub_dict and num_start > 0
            ):
                num_start -= 1 # After the loop ends, num_start is the maximum number that we can find, which is less than or equal to l and has been cached.
        else: # If this pairing has never been computed before, initialize an empty sub-cache.
            self.Cache_H[(target_c, other_c)] = dict()
            num_start = 0 # And set num_start to 0, indicating that computation must start from scratch.
        
        # --- 2. Initialize computation starting point H ---
        if num_start == 0: # If num_start is 0, it means no cache is available and the initial state needs to be created.
            H = dict()
            H_config = np.zeros(
                self.c1_type_shape, dtype=np.uint8
            )
            H_config = HashableArrayWrapper(H_config)
            H[(target_c, H_config)] = Rational(
                1, 1
            )
        else: # If a cache is found, directly load the result corresponding to num_start as the starting point for computation.
            H = self.Cache_H[(target_c, other_c)][num_start]
        
        # --- 3. Iterative computation ---
        for j in range(num_start + 1, l + 1): # This loop starts from the cached num_start step and calculates step by step until reaching the required number l.
            H_new = defaultdict(
                lambda: Rational(0, 1)
            ) # H_new is used to store the new states generated in this iteration (i.e., pairing with the j-th element).
            for (target_c_old, H_config_old), W in H.items(): # Iterate over all result states from the previous step (paired with j-1 elements).
                for (target_c_new, other_c_new), rij in self.t_update_dict[
                    (target_c_old, other_c) # The t_update_dict stores all possible transitions of target_c_old paired with an other_c, and rij represents the weight of this transition.
                ].items():
                    H_config_new = np.array(H_config_old.array)
                    H_config_new[other_c_new] += 1 # other_c_new is the new state of other_c. Increment the count of elements at the index corresponding to the new state in the dictionary.
                    H_config_new = HashableArrayWrapper(
                        H_config_new
                    )
                    H_new[(target_c_new, H_config_new)] += (
                        W * rij
                    ) # New weight = previous weight W * transition weight rij.
            H = H_new # Overwrite H with the newly computed states H_new as the starting point for the next iteration.
            self.Cache_H[(target_c, other_c)][j] = H # Store the result of this iteration (i.e., paired with j elements) in the cache for future use.
        return H


class HashableArrayWrapper(object):
    def __init__(self, input_array: np.ndarray):
        self.array = input_array.astype(np.uint8, copy=False) # Convert the input NumPy array to uint8 type and store it.

    def __hash__(self):
        return int(hashlib.sha1(self.array).hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return int(hashlib.sha1(self.array).hexdigest(), 16) == int(
                hashlib.sha1(other.array).hexdigest(), 16
            )
        return False

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return f"HashableArrayWrapper({self.array})"
