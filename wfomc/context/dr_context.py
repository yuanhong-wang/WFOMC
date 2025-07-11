from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
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

        self.cnt_remainder = []  # r       (int)
        self.mod_pred_index = []
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

    def bulid_wij(self, r, n_cells, domain_size):
        """
        Build the state transition weight table

        Args:
            self.cnt_preds (list): A list of counting predicate names.
            self.cnt_params (dict): A list of parameters associated with each predicate.
            self.mod_pred_index (list): A list indicate which predicate has mod.

        Returns:
            defaultdict: State transition weight table t_updates
        """
        t_updates = defaultdict(lambda: defaultdict(
            lambda: Rational(0, 1)))  # Initialize a two-level defaultdict: outer key is (c1, c2) (current joint state); inner key is (c1_new, c2_new) (next joint state); value is transition weight

        # construct c-type below

        if self.exist_mod:
            final_list = [tuple(range(2)) for _ in self.ext_preds]  # do not process exist predicates
            for idx, k in enumerate(self.cnt_params):
                if idx in self.mod_pred_index:
                    final_k = (domain_size // k) * k  # mod 2 -> 2,4,6...
                    final_list += [tuple(range(final_k + 1))]
                    # print("total_k-->>",total_k)
                else:
                    final_list += [tuple(range(k + 1))]
                # print("final_list-->>",final_list)
            all_ts = list(product(*(final_list)))
            # print("all_ts-->>",all_ts)
        else:
            # Enumerate all valid internal states for a single cell:
            # 1. tuple(range(2)) gives value set {0,1} for each exist predicate;
            # 2. tuple(range(k+1)) gives {0,…,k} for each counting predicate;
            # 3. Concatenate these lists and do Cartesian product (itertools.product) to get all combinations  all_ts like [(b1,…,bn, c1,…,cm), …]。
            all_ts = list(
                product(*([tuple(range(2)) for _ in self.ext_preds] + [tuple(range(i + 1)) for i in self.cnt_params]))
            )
        # Double loop through ordered cell pairs (i,j) (allowing i=j, same cell pair)
        for i in range(n_cells):
            for j in range(n_cells):
                for t1 in all_ts:  # Enumerate source cell i's current state t1 and target cell j's current state t2
                    for t2 in all_ts:
                        # Traverse all state increments registered in relation dict r[(i,j)]
                        # • dt applied to t1
                        # • reverse_dt applied to t2
                        # • rijt is the weight (probability/contribution) of this increment
                        for (dt, reverse_dt), rijt in r[(i, j)].items():
                            # 1) First do "old-style subtraction" once
                            t1_new = [a - b for a, b in zip(t1, dt)]
                            t2_new = [a - b for a, b in zip(t2, reverse_dt)]

                            # === 2) For counting predicates, distinguish two semantics ===
                            if self.exist_mod:
                                for idx, k_i in enumerate(self.cnt_params, start=len(self.ext_preds)):
                                    if idx in self.mod_pred_index:
                                        pass
                                        if t1_new[idx] == -1:
                                            t1_new[idx] = t1_new[idx] % k_i
                                        if t2_new[idx] == -1:
                                            t2_new[idx] = t1_new[idx] % k_i
                                    else:  # # If current counting predicate index is not in self.mod_pred_index, handle normally
                                        if t1_new[len(self.ext_preds) + idx] == -1 or t2_new[len(self.ext_preds) + idx] == -1:  # If a negative number appears, it is illegal and skip
                                            continue
                            else:  # Only normal counting quantifiers ∃=k (no mod-k): negative numbers are invalid
                                if any(
                                        t1_new[len(self.ext_preds) + p] < 0 or
                                        t2_new[len(self.ext_preds) + p] < 0
                                        for p in range(len(self.cnt_params))
                                ):
                                    continue

                            # Fix boolean states
                            for idx in range(
                                    len(self.ext_preds)):  # Existential quantifier predicate states (boolean) must be non-negative: e.g., t1_new = [1, -1, 1] → fixed to [1, 0, 1]
                                t1_new[idx] = max(t1_new[idx], 0)
                                t2_new[idx] = max(t2_new[idx], 0)

                            # write to t_updates
                            c1 = (i,) + t1  # Current source cell state (i, t1)
                            c2 = (j,) + t2  # Current target cell state (j, t2)
                            c1_new = (i,) + tuple(t1_new)  # New source cell state
                            c2_new = (j,) + tuple(t2_new)  # New target cell state
                            t_updates[(c1, c2)][(c1_new, c2_new)] += rijt
        return t_updates

    def build_weight(self, cells, cell_graph):
        """
        Build weight and relation dictionaries r

        Args:
            cells: List of cell types
            self.ext_preds: List of existential quantifier predicates
            self.cnt_preds: List of counting predicates
            self.cnt_params: List of counting parameters
            self.binary_evidence: List of binary evidence
            cell_graph: Cell graph object

        Returns:
            tuple: (w2t dictionary, w weight dictionary, r relation dictionary)
        """

        n_cells = len(cells)
        w2t = dict()  # Mapping from cell index to predicate states dictionary.
        w = defaultdict(lambda: Rational(0, 1))  # Weight dictionary for each cell type.
        r = defaultdict(lambda: defaultdict(lambda: Rational(0, 1)))  # Relation dictionary between cell pairs.
        for i in range(n_cells):
            cell_weight = cell_graph.get_cell_weight(cells[i])
            t = list()  # Binary list representing predicate states (1=true, 0=false)
            for pred in self.ext_preds:  # For boolean predicates (self.ext_preds), store 1 if positive, 0 if negative
                if cells[i].is_positive(pred):
                    t.append(1)
                else:
                    t.append(0)
            for idx, (pred, param) in enumerate(zip(self.cnt_preds, self.cnt_params)):  # For counting predicates, store param-1 if positive, param if negative
                if cells[i].is_positive(pred):
                    if self.exist_mod and idx in self.mod_pred_index:
                        t.append(self.cnt_remainder[idx] - 1)  # If this counting predicate is mod-k, store the self.cnt_remainder
                    else:
                        t.append(param - 1)
                else:
                    if self.exist_mod and idx in self.mod_pred_index:
                        t.append(self.cnt_remainder[idx])  # If this counting predicate is mod-k, store the self.cnt_remainder
                    else:
                        t.append(param)
            w2t[i] = tuple(t)  # The number that needs to be fulfilled
            w[i] = w[i] + cell_weight  # Accumulate cell weight

            for j in range(n_cells):  # Iterate through all cell pairs and binary evidence
                cell1 = cells[i]
                cell2 = cells[j]
                for evi_idx, evidence in enumerate(self.binary_evidence):  # Iterate through all possible binary evidence
                    t = list()  # t: Store the "positive" state (a→b direction) of each predicate in the current evidence
                    reverse_t = list()  # reverse_t: Stores the "reverse" state (b→a direction) of each predicate
                    two_table_weight = cell_graph.get_two_table_weight(
                        (cell1, cell2), evidence
                    )
                    if two_table_weight == Rational(0, 1):  # Skip if weight is zero (invalid configuration)
                        continue
                    for pred_idx, pred in enumerate(self.ext_preds + self.cnt_preds):  # Check predicate states for both directions
                        if (evi_idx >> (2 * pred_idx)) & 1 == 1:
                            reverse_t.append(1)
                        else:
                            reverse_t.append(0)
                        if (evi_idx >> (2 * pred_idx + 1)) & 1 == 1:
                            t.append(1)
                        else:
                            t.append(0)
                    r[(i, j)][(tuple(t), tuple(reverse_t))] = two_table_weight  # Store relation weight with predicate state combinations
        return w2t, w, r

    def build_c_type_shape(self):
        self.c_type_shape = list(2 for _ in self.ext_preds)
        for idx, k in enumerate(self.cnt_params):
            if idx in self.mod_pred_index:  # This quantifier is of the type ∃_{r mod k}
                self.c_type_shape.append(k)  # 0 to k-1
            else:  # 是∃_{k}
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
                self.mod_pred_index.append(idx)

                # cnt_param_raw 是 (r, k)
                r, k = cnt_param_raw
                self.cnt_remainder.append(r)
                self.cnt_params.append(k)  # Just put k into cnt_params
            else:
                self.self.cnt_remainder.append(None)
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

class ConfigUpdater:
    def __init__(self, t_updates, shape, cache):
        self.t_updates = t_updates
        self.shape = shape
        self.Cache_F = cache  # global Cache_F

    def update_config(self, target_c, other_c, l):
        if (target_c, other_c) in self.Cache_F:  # Check if we have already computed this cell pair in cache.
            config_updates_cache_num = self.Cache_F[(target_c, other_c)]
            num_start = l
            while num_start not in config_updates_cache_num and num_start > 0:  # Find the largest cached j ≤ l
                num_start -= 1
        else:  # # Initialize cache for this cell pair if not exists
            self.Cache_F[(target_c, other_c)] = dict()
            num_start = 0
        # Initialize F
        if num_start == 0:
            F = dict()
            u_config = np.zeros(self.shape, dtype=np.uint8)
            u_config = HashableArrayWrapper(u_config)
            F[(target_c, u_config)] = Rational(1, 1)  # Initial weight is 1 for target_c with no configuration changes
        else:
            F = self.Cache_F[(target_c, other_c)][num_start]
        # Main loop: j from num_start+1 to l
        for j in range(num_start + 1, l + 1):
            F_new = defaultdict(lambda: Rational(0, 1))  # Create new dictionary for this iteration
            for (target_c_old, u), W in F.items():  # Process each existing state transition
                for (target_c_new, other_c_new), rij in self.t_updates[(target_c_old, other_c)].items():
                    F_config_new = np.array(u.array)
                    F_config_new[other_c_new] += 1
                    F_config_new = HashableArrayWrapper(F_config_new)
                    F_new[(target_c_new, F_config_new)] += W * rij
            F = F_new
            self.Cache_F[(target_c, other_c)][j] = F  # Cache results for this iteration
        return F

class HashableArrayWrapper(object):
    def __init__(self, input_array: np.ndarray):
        self.array = input_array.astype(np.uint8, copy=False)

    def __hash__(self):
        return int(hashlib.sha1(self.array).hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return int(hashlib.sha1(self.array).hexdigest(), 16) == \
                int(hashlib.sha1(other.array).hexdigest(), 16)
        return False

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return f"HashableArrayWrapper({self.array})"