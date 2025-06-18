import numpy as np
import hashlib

from logzero import logger
from collections import Counter, defaultdict
from itertools import product
from wfomc.cell_graph.cell_graph import build_cell_graphs
from wfomc.fol.sc2 import SC2
from wfomc.fol.syntax import AUXILIARY_PRED_NAME, X, Y, AtomicFormula, Const, Pred, QFFormula, top, a, b
from wfomc.fol.utils import new_predicate
from wfomc.network.constraint import CardinalityConstraint
from wfomc.problems import WFOMCProblem
from wfomc.utils import multinomial, MultinomialCoefficients
from wfomc.utils.polynomial import Rational
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


def preprocess(sentence: SC2):
    uni_formula = sentence.uni_formula
    while(not isinstance(uni_formula, QFFormula)):
        uni_formula = uni_formula.quantified_formula
    ext_formulas = sentence.ext_formulas
    cnt_formulas = sentence.cnt_formulas
    # add auxiliary predicates for existential and counting quantified formulas
    ext_preds = list()
    cnt_preds = list()
    cnt_params = list()
    for formula in ext_formulas:
        # NOTE: assume all existential formulas are of the form VxEy
        qf_formula = formula.quantified_formula.quantified_formula
        aux_pred = new_predicate(2, AUXILIARY_PRED_NAME)
        uni_formula = uni_formula & qf_formula.equivalent(aux_pred(X, Y))
        ext_preds.append(aux_pred)

    for formula in cnt_formulas:
        # NOTE: assume all counting formulas are of the form VxE=k y: f(x,y)
        qf_formula = formula.quantified_formula.quantified_formula
        cnt_param = formula.quantified_formula.quantifier_scope.count_param
        cnt_params.append(cnt_param)
        aux_pred = new_predicate(2, AUXILIARY_PRED_NAME)
        uni_formula = uni_formula & qf_formula.equivalent(aux_pred(X, Y))
        cnt_preds.append(aux_pred)
    return uni_formula, ext_preds, cnt_preds, cnt_params


def domain_recursive_wfomc(problem: WFOMCProblem) -> RingElement:
    domain: set[Const] = problem.domain
    sentence: SC2 = problem.sentence
    weights: dict[Pred, tuple[Rational, Rational]] = problem.weights
    def get_weight(pred: Pred):
        if pred in weights:
            return weights[pred]
        return (Rational(1, 1), Rational(1, 1))
    cardinality_constraint: CardinalityConstraint = problem.cardinality_constraint
    formula, ext_preds, cnt_preds, cnt_params = preprocess(sentence)
    logger.info('Universally quantified formula: %s', formula)
    logger.info('Existential predicates: %s', ext_preds)
    logger.info('Counting predicates: %s', cnt_preds)
    logger.info('Counting parameters: %s', cnt_params)

    # the index of binary evidence has its specific meaning
    # every index (in binary system) is split into parts of length 2,
    # each part corresponds to the evidence of a predicate:
    # 00: means (~R(a,b), ~R(b,a))
    # 01: means (~R(a,b), R(b,a))
    # 10: means (R(a,b), ~R(b,a))
    # 11: means (R(a,b), R(b,a))
    # e.g., ext_preds = [R1, R2], cnt_preds = [R3], then
    # 000000 means (~R1(a, b), ~R1(b, a), ~R2(a, b), ~R2(b, a), ~R3(a, b), ~R3(b, a))
    # 000001 means (~R1(a, b), ~R1(b, a), ~R2(a, b), ~R2(b, a), ~R3(a, b), R3(b, a))
    # 000010 means (~R1(a, b), ~R1(b, a), ~R2(a, b), ~R2(b, a), R3(a, b), ~R3(b, a))
    # 000011 means (~R1(a, b), ~R1(b, a), ~R2(a, b), ~R2(b, a), R3(a, b), R3(b, a))
    # 000100 means (~R1(a, b), ~R1(b, a), ~R2(a, b), R2(b, a), ~R3(a, b), ~R3(b, a))
    # 000101 means (~R1(a, b), ~R1(b, a), ~R2(a, b), R2(b, a), ~R3(a, b), R3(b, a))
    binary_evidence = list()
    ext_atoms = list(
        ((~pred(a, b), ~pred(b, a)),
         (~pred(a, b), pred(b, a)),
         (pred(a, b), ~pred(b, a)),
         (pred(a, b), pred(b, a)))
        for pred in ext_preds[::-1])
    cnt_atoms = list(
        ((~pred(a, b), ~pred(b, a)),
         (~pred(a, b), pred(b, a)),
         (pred(a, b), ~pred(b, a)),
         (pred(a, b), pred(b, a)))
        for pred in cnt_preds[::-1])
    for atoms in product(*cnt_atoms, *ext_atoms):
        binary_evidence.append(frozenset(sum(atoms, start = ())))

    res = Rational(0, 1)
    domain_size = len(domain)
    MultinomialCoefficients.setup(domain_size)
    for cell_graph, graph_weight in build_cell_graphs(formula, get_weight):
        res_ = Rational(0, 1)
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        w2t = dict()
        w = defaultdict(lambda: Rational(0, 1))
        r = defaultdict(lambda: defaultdict(lambda: Rational(0, 1)))
        for i in range(n_cells):
            cell_weight = cell_graph.get_cell_weight(cells[i])
            t = list()
            for pred in ext_preds:
                if cells[i].is_positive(pred):
                    t.append(0)
                else:
                    t.append(1)
            for pred, param in zip(cnt_preds, cnt_params):
                if cells[i].is_positive(pred):
                    t.append(param - 1)
                else:
                    t.append(param)
            w2t[i] = tuple(t)
            w[i] = w[i] + cell_weight
            for j in range(n_cells):
                cell1 = cells[i]
                cell2 = cells[j]
                for evi_idx, evidence in enumerate(binary_evidence):
                    t = list()
                    reverse_t = list()
                    two_table_weight = cell_graph.get_two_table_weight(
                        (cell1, cell2), evidence
                    )
                    if two_table_weight == Rational(0, 1):
                        continue
                    for pred_idx, pred in enumerate(ext_preds + cnt_preds):
                        if (evi_idx >> (2 * pred_idx)) & 1 == 1:
                            reverse_t.append(1)
                        else:
                            reverse_t.append(0)
                        if (evi_idx >> (2 * pred_idx + 1)) & 1 == 1:
                            t.append(1)
                        else:
                            t.append(0)
                    r[(i, j)][(tuple(t), tuple(reverse_t))] = two_table_weight

        t_updates = defaultdict(lambda: defaultdict(lambda: Rational(0, 1)))
        all_ts = list(
            product(*([tuple(range(2)) for _ in ext_preds] + [tuple(range(k + 1)) for k in cnt_params]))
        )
        for i in range(n_cells):
            for j in range(n_cells):
                for t1 in all_ts:
                    for t2 in all_ts:
                        for (dt, reverse_dt), rijt in r[(i, j)].items():
                            t1_new = list(i - j for i, j in zip(t1, dt))
                            t2_new = list(i - j for i, j in zip(t2, reverse_dt))
                            # print(t1, t2, t1_new, t2_new, rijt)
                            if any(
                                t1_new[idx + len(ext_preds)] < 0 or \
                                t2_new[idx + len(ext_preds)] < 0
                                for idx, _ in enumerate(cnt_params)
                            ):
                                continue
                            for idx in range(len(ext_preds)):
                                t1_new[idx] = max(t1_new[idx], 0)
                                t2_new[idx] = max(t2_new[idx], 0)
                            c1 = (i, ) + t1
                            c2 = (j, ) + t2
                            c1_new = (i, ) + tuple(t1_new)
                            c2_new = (j, ) + tuple(t2_new)
                            t_updates[(c1, c2)][(c1_new, c2_new)] += rijt
        # print(all_ts)
        # print(w)

        shape = (n_cells, ) + tuple(2 for _ in ext_preds) + tuple(k + 1 for k in cnt_params)

        config_updates_cache = dict()
        def update_config(target_c, other_c, num):
            if (target_c, other_c) in config_updates_cache:
                config_updates_cache_num = config_updates_cache[(target_c, other_c)]
                start_num = num
                while start_num not in config_updates_cache_num and start_num > 0:
                    start_num -= 1
            else:
                config_updates_cache[(target_c, other_c)] = dict()
                start_num = 0

            if start_num == 0:
                F = dict()
                F_config = np.zeros(shape, dtype=np.uint8)
                F_config = HashableArrayWrapper(F_config)
                F[(target_c, F_config)] = Rational(1, 1)
            else:
                F = config_updates_cache[(target_c, other_c)][start_num]

            for j in range(start_num + 1, num + 1):
                F_new = defaultdict(lambda: Rational(0, 1))
                for (target_c_old, F_config_old), V in F.items():
                    for (target_c_new, other_c_new), rij in t_updates[(target_c_old, other_c)].items():
                        F_config_new = np.array(F_config_old.array)
                        F_config_new[other_c_new] += 1
                        F_config_new = HashableArrayWrapper(F_config_new)
                        F_new[(target_c_new, F_config_new)] += V * rij
                F = F_new
                config_updates_cache[(target_c, other_c)][j] = F
            return F

        Cache = dict()
        def domain_recursion(config):
            if config in Cache:
                return Cache[config]
            if config.array.sum() == 0:
                return Rational(1, 1)
            T = defaultdict(lambda: Rational(0, 1))
            new_config = HashableArrayWrapper(
                np.array(config.array, copy=True, dtype=np.uint8)
            )
            target_c = tuple(np.argwhere(new_config.array > 0)[-1])
            new_config.array[target_c] -= 1
            # init_t = w2t[indices[0]]
            # init_t = list(i - j for i, j in zip(indices[1:], target_t))
            # for i in range(len(ext_preds)):
            #     init_t[i] = max(init_t[i], 0)
            # for i in range(len(cnt_preds)):
            #     if init_t[i + len(ext_preds)] < 0:
            #         return Rational(0, 1)
            # init_t = tuple(init_t)
            # target_c = (indices[0], ) + init_t
            F = dict()
            F_config = np.zeros(shape, dtype=np.uint8)
            F_config = HashableArrayWrapper(F_config)
            F[(target_c, F_config)] = Rational(1, 1)
            for other_c in np.argwhere(new_config.array > 0):
                other_c = tuple(other_c.flatten())
                F_new = defaultdict(lambda: Rational(0, 1))
                num = new_config.array[other_c]
                for (target_c, F_config), V in F.items():
                    # print(other_c, target_c, num)
                    F_update = update_config(target_c, other_c, num)
                    # print(F_update)
                    for target_c_new, F_config_update in F_update.keys():
                        F_config_new = HashableArrayWrapper(
                            F_config.array + F_config_update.array
                        )
                        F_new[(target_c_new, F_config_new)] += V * F_update[(target_c_new, F_config_update)]
                F = F_new
            # print(F)
            for (last_target_c, last_F_config), V in F.items():
                if all(i == 0 for i in last_target_c[1:]):
                    T[last_F_config] += V
            # print(T)
            ret = Rational(0, 1)
            for recursive_config, weight in T.items():
                W = domain_recursion(recursive_config)
                ret = ret + (weight * W)
            Cache[config] = ret
            return ret

        for config in multinomial(n_cells, domain_size):
            init_config = np.zeros(shape, dtype=np.uint8)
            W = Rational(1, 1)
            for i, n in enumerate(config):
                init_config[(i, ) + w2t[i]] = n
                W = W * (w[i] ** n)
            init_config = HashableArrayWrapper(init_config)
            dr_res = domain_recursion(init_config)
            res += (MultinomialCoefficients.coef(config) * W *
                    dr_res * graph_weight)
        # print(Cache)
    return res
