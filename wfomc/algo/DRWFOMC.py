import numpy as np
import hashlib

from logzero import logger
from collections import Counter, defaultdict
from itertools import product
from wfomc.cell_graph.cell_graph import build_cell_graphs
from wfomc.context.dr_context import DRWFOMCContext
from wfomc.fol.sc2 import SC2
from wfomc.fol.syntax import AUXILIARY_PRED_NAME, X, Y, AtomicFormula, Const, Pred, QFFormula, top, a, b
from wfomc.fol.utils import new_predicate
from wfomc.network.constraint import CardinalityConstraint
from wfomc.problems import WFOMCProblem
from wfomc.utils import multinomial, MultinomialCoefficients
from wfomc.utils.polynomial import Rational, coeff_dict, expand
from wfomc.utils.third_typing import RingElement


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


# Global variable to indicate if there is a mod-k counting quantifier

def build_weight(cells, ext_preds, cnt_preds, cnt_params, binary_evidence, cell_graph, exist_mod=False, cnt_remainder=None, mod_preds_index=None):
    """
    Build weight and relation dictionaries r

    Args:
        cells: List of cell types
        ext_preds: List of existential quantifier predicates
        cnt_preds: List of counting predicates
        cnt_params: List of counting parameters
        binary_evidence: List of binary evidence
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
        for pred in ext_preds:  # For boolean predicates (ext_preds), store 1 if positive, 0 if negative
            if cells[i].is_positive(pred):
                t.append(1)
            else:
                t.append(0)
        for idx, (pred, param) in enumerate(zip(cnt_preds, cnt_params)):  # For counting predicates, store param-1 if positive, param if negative
            if cells[i].is_positive(pred):
                if exist_mod and idx in mod_preds_index:
                    t.append(cnt_remainder[idx] - 1)  # If this counting predicate is mod-k, store the remainder
                else:
                    t.append(param - 1)
            else:
                if exist_mod and idx in mod_preds_index:
                    t.append(cnt_remainder[idx])  # If this counting predicate is mod-k, store the remainder
                else:
                    t.append(param)
        w2t[i] = tuple(t)  # The number that needs to be fulfilled
        w[i] = w[i] + cell_weight  # Accumulate cell weight

        for j in range(n_cells):  # Iterate through all cell pairs and binary evidence
            cell1 = cells[i]
            cell2 = cells[j]
            for evi_idx, evidence in enumerate(binary_evidence):  # Iterate through all possible binary evidence
                t = list()  # t: Store the "positive" state (a→b direction) of each predicate in the current evidence
                reverse_t = list()  # reverse_t: Stores the "reverse" state (b→a direction) of each predicate
                two_table_weight = cell_graph.get_two_table_weight(
                    (cell1, cell2), evidence
                )
                if two_table_weight == Rational(0, 1):  # Skip if weight is zero (invalid configuration)
                    continue
                for pred_idx, pred in enumerate(ext_preds + cnt_preds):  # Check predicate states for both directions
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


def bulid_wij(r, ext_preds, cnt_preds, cnt_params, n_cells, mod_pred_index, exist_mod=False):
    """
    Build the state transition weight table

    Args:
        cnt_preds (list): A list of counting predicate names.
        cnt_params (dict): A list of parameters associated with each predicate.
        mod_pred_index (list): A list indicate which predicate has mod.

    Returns:
        defaultdict: State transition weight table t_updates
    """
    t_updates = defaultdict(lambda: defaultdict(
        lambda: Rational(0, 1)))  # Initialize a two-level defaultdict: outer key is (c1, c2) (current joint state); inner key is (c1_new, c2_new) (next joint state); value is transition weight
    global domain_size

    # construct c-type below

    if exist_mod:
        final_list = [tuple(range(2)) for _ in ext_preds]  # do not process exist predicates
        for idx, k in enumerate(cnt_params):
            if idx in mod_pred_index:
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
            product(*([tuple(range(2)) for _ in ext_preds] + [tuple(range(i + 1)) for i in cnt_params]))
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
                        if exist_mod:
                            for idx, k_i in enumerate(cnt_params, start=len(ext_preds)):
                                if idx in mod_pred_index:
                                    pass
                                    if t1_new[idx] == -1:
                                        t1_new[idx] = t1_new[idx] % k_i
                                    if t2_new[idx] == -1:
                                        t2_new[idx] = t1_new[idx] % k_i
                                else:  # # If current counting predicate index is not in mod_pred_index, handle normally
                                    if t1_new[len(ext_preds) + idx] == -1 or t2_new[len(ext_preds) + idx] == -1:  # If a negative number appears, it is illegal and skip
                                        continue
                        else:  # Only normal counting quantifiers ∃=k (no mod-k): negative numbers are invalid
                            if any(
                                    t1_new[len(ext_preds) + p] < 0 or
                                    t2_new[len(ext_preds) + p] < 0
                                    for p in range(len(cnt_params))
                            ):
                                continue

                        # Fix boolean states
                        for idx in range(
                                len(ext_preds)):  # Existential quantifier predicate states (boolean) must be non-negative: e.g., t1_new = [1, -1, 1] → fixed to [1, 0, 1]
                            t1_new[idx] = max(t1_new[idx], 0)
                            t2_new[idx] = max(t2_new[idx], 0)

                        # write to t_updates
                        c1 = (i,) + t1  # Current source cell state (i, t1)
                        c2 = (j,) + t2  # Current target cell state (j, t2)
                        c1_new = (i,) + tuple(t1_new)  # New source cell state
                        c2_new = (j,) + tuple(t2_new)  # New target cell state
                        t_updates[(c1, c2)][(c1_new, c2_new)] += rijt
    return t_updates


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


def domain_recursive_wfomc(context: DRWFOMCContext) -> RingElement:
    domain: set[Const] = context.domain
    sentence: SC2 = context.sentence
    weights: dict[Pred, tuple[Rational, Rational]] = context.weights
    get_weight = context.get_weight
    leq_pred: Pred = context.leq_pred
    cardinality_constraint: CardinalityConstraint = context.cardinality_constraint
    ##
    formula = context.uni_formula
    ext_preds = context.ext_preds
    cnt_preds = context.cnt_preds
    cnt_params = context.cnt_params
    cnt_remainder = context.remainder

    binary_evidence = context.binary_evidence

    mod_preds_index = context.mod_preds_index
    exist_mod = context.exist_mod
    c_type_shape = context.c_type_shape

    result = Rational(0, 1)
    global domain_size
    domain_size = len(domain)
    MultinomialCoefficients.setup(domain_size)
    for cell_graph, graph_weight in build_cell_graphs(formula, get_weight, leq_pred):
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        w2t, w, r = build_weight(cells, ext_preds, cnt_preds, cnt_params, binary_evidence, cell_graph, exist_mod, cnt_remainder, mod_preds_index)  # build_weight
        t_updates = bulid_wij(r, ext_preds, cnt_preds, cnt_params, n_cells, mod_preds_index, exist_mod)  # t_updates
        # shape = (n_cells,) + tuple(2 for _ in ext_preds) + tuple(k + 1 for k in cnt_params)
        shape = (n_cells,) + tuple(c_type_shape)

        Cache_F = dict()  # Global Cache_F
        config_updater = ConfigUpdater(t_updates, shape, Cache_F)  # update_config
        update_config = config_updater.update_config  # this is a function

        Cache_T = dict()  # key = (config)

        # core of recursion
        def domain_recursion(config):
            if config in Cache_T:
                return Cache_T[config]

            if config.array.sum() == 0:  # All elements have been consumed
                return Rational(1, 1)

            T = defaultdict(lambda: Rational(0, 1))
            new_config = HashableArrayWrapper(  # Create a copy of current configuration (to avoid modifying original)
                np.array(config.array, copy=True, dtype=np.uint8)
            )
            # select element
            target_c = tuple(np.argwhere(new_config.array > 0)[-1])  # Select last non-zero cell as target (processing order strategy)
            new_config.array[target_c] -= 1  # Remove this element
            F = dict()
            u_config = np.zeros(shape, dtype=np.uint8)
            u_config = HashableArrayWrapper(u_config)
            F[(target_c, u_config)] = Rational(1, 1)

            # Pair with "other elements" in sequence (outer loop)
            for other_c in np.argwhere(new_config.array > 0):
                other_c = tuple(other_c.flatten())  # Get cell coordinates
                F_new = defaultdict(lambda: Rational(0, 1))
                l = new_config.array[other_c]  # How many elements of this type remain
                # Expand F by calling update_config (inner core)
                for (target_c, u_config), W in F.items():
                    F_update = update_config(target_c, other_c, l)  # Calculate all possible results after target_c interacts with other_c for num times
                    for target_c_new, u_config_update in F_update.keys():  # Merge transition results
                        F_config_new = HashableArrayWrapper(
                            u_config.array + u_config_update.array
                        )
                        F_new[(target_c_new, F_config_new)] += W * F_update[(target_c_new, u_config_update)]
                F = F_new  # Update current transition path set

            # # Filter "target satisfied" terminal states → T
            for (last_target_c,
                 last_F_config), W in F.items():  # Among all the pairing methods, only retain the portion where the target elements have fully met the constraints, and merge their weights into the configuration T of the next layer of recursion.
                if all(i == 0 for i in last_target_c[1:]):  # # Check the predicate status. Filter out valid results: The predicate status of target_c must be all zeros.
                    T[last_F_config] += W

            # Recursively calculate weighted sum of all subproblems
            ret = Rational(0, 1)
            for recursive_config, weight in T.items():
                W = domain_recursion(recursive_config)  # Recursive call to subconfiguration
                ret = ret + (weight * W)
            Cache_T[config] = ret  # Cache current configuration result
            return ret

        # main code
        for config in multinomial(n_cells, domain_size):
            init_config = np.zeros(shape, dtype=np.uint8)
            W = Rational(1, 1)
            for i, n in enumerate(config):
                init_config[(i,) + w2t[i]] = n  # i is cell index, n is element num of cell i
                W = W * (w[i] ** n)
            init_config = HashableArrayWrapper(init_config)
            dr_res = domain_recursion(init_config)  # call recursison
            result += (MultinomialCoefficients.coef(config) * W * dr_res * graph_weight)
    return result
