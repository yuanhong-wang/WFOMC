import numpy as np
import hashlib

from logzero import logger
from collections import defaultdict
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
        for pred in ext_preds)
    cnt_atoms = list(
        ((~pred(a, b), ~pred(b, a)),
         (~pred(a, b), pred(b, a)),
         (pred(a, b), ~pred(b, a)),
         (pred(a, b), pred(b, a)))
        for pred in cnt_preds)
    for atoms in product(*ext_atoms, *cnt_atoms):
        binary_evidence.append(frozenset(sum(atoms, start = ())))

    def domain_recursion(config):
        if config.sum() == 1:
            return Rational(1, 1)
        chosen_element = tuple(
            i[0] for i in config.nonzero()
        )
        config[chosen_element] -= 1
        ret = Rational(0, 1)
        for two_table_config in product(
            *list(
                multinomial(len(binary_evidence), num)
                for num in config.flatten()
            )
        ):
            two_table_config = np.array(two_table_config, dtype=int).reshape(
                config.shape + (len(binary_evidence), )
            )
            chosen_element_two_table_config = two_table_config.sum(
                axis=tuple(range(two_table_config.ndim - 1))
            )
            pred_config = list(0 for _ in range(len(ext_preds) + len(cnt_preds)))
            for i in range(len(ext_preds) + len(cnt_preds)):
                for j, num in enumerate(chosen_element_two_table_config):
                    if (j >> (2 * i + 1)) & 1 == 1:
                        pred_config[i] += num
            if any(
                chosen_element[idx + 1] == 1 and pred_config[idx] == 0
                for idx in range(len(ext_preds))
            ):
                continue
            if any(
                chosen_element[idx + len(ext_preds) + 1] != pred_config[idx]
                for idx in range(len(cnt_preds))
            ):
                continue
            new_config = np.zeros(config.shape, dtype=int)
            for indices, _ in np.ndenumerate(config):
                other_config = two_table_config[indices]
                for i, num in enumerate(other_config):
                    new_indices = list(indices)
                    for j in range(len(ext_preds)):
                        if (i >> (2 * j)) & 1 == 1:
                            new_indices[1 + j] = 0
                    for j in range(len(cnt_preds)):
                        if (i >> (2 * (j + len(ext_preds)))) & 1 == 1:
                            new_indices[1 + len(ext_preds) + j] -= 1
                    if all(idx >= 0 for idx in new_indices):
                        new_config[tuple(new_indices)] += num
            if new_config.sum() == config.sum():
                ret_dp = domain_recursion(new_config)
                ret = ret + ret_dp
        return ret


    res = Rational(0, 1)
    domain_size = len(domain)
    MultinomialCoefficients.setup(domain_size)
    for cell_graph, graph_weight in build_cell_graphs(formula, get_weight):
        res_ = Rational(0, 1)
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        w = defaultdict(lambda: Rational(0, 1))
        r = defaultdict(lambda: defaultdict(lambda: Rational(0, 1)))
        for i in range(n_cells):
            cell_weight = cell_graph.get_cell_weight(cells[i])
            t = list()
            for pred in ext_preds + cnt_preds:
                if cells[i].is_positive(pred):
                    t.append(1)
                else:
                    t.append(0)
            w[(i, tuple(t))] = w[(i, tuple(t))] + cell_weight
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
        for (i, w_i_t), _ in w.items():
            for (j, w_j_t), _ in w.items():
                for t1 in all_ts:
                    for t2 in all_ts:
                        if any(t1[idx] < w_i_t[idx] for idx in range(len(t1))) or \
                                any(t2[idx] < w_j_t[idx] for idx in range(len(t2))):
                            continue
                        for (dt, reverse_dt), rijt in r[(i, j)].items():
                            t1_new = list(i + j for i, j in zip(t1, dt))
                            t2_new = list(i + j for i, j in zip(t2, reverse_dt))
                            if any(
                                t1_new[idx + len(ext_preds)] > param or \
                                t2_new[idx + len(ext_preds)] > param
                                for idx, param in enumerate(cnt_params)
                            ):
                                continue
                            for idx in range(len(ext_preds)):
                                t1_new[idx] = min(t1_new[idx], 1)
                                t2_new[idx] = min(t2_new[idx], 1)
                            c1 = (i, ) + t1
                            c2 = (j, ) + t2
                            c1_new = (i, ) + tuple(t1_new)
                            c2_new = (j, ) + tuple(t2_new)
                            t_updates[(c1, c2)][(c1_new, c2_new)] += rijt
        # print(all_ts)
        print(w)
        # print(t_updates)

        T = dict()
        shape = (n_cells, ) + tuple(2 for _ in ext_preds) + tuple(k + 1 for k in cnt_params)
        for (i, t), weight in w.items():
            c = (i, ) + t
            config = np.zeros(shape, dtype=np.uint8)
            config[c] = 1
            config = HashableArrayWrapper(config)
            T[config] = weight
        print('init T:')
        print(T)

        all_T_sources = list()
        for h in range(2, domain_size + 1):
            print(f'------------------------ h = {h} ------------------------')
            T_new = defaultdict(lambda: Rational(0, 1))
            T_sources = defaultdict(lambda: set())
            for (i, target_t), w_weight in w.items():
                for config, W in T.items():
                    F = dict()
                    F_config = np.zeros(shape, dtype=np.uint8)
                    F_config = HashableArrayWrapper(F_config)
                    F[((i, ) + target_t, F_config)] = w_weight

                    config_tmp = np.array(config.array)
                    for j in range(h - 1):
                        F_new = defaultdict(lambda: Rational(0, 1))
                        # get the first index whose value > 0
                        c_j = tuple(
                            idx[0] for idx in np.nonzero(config_tmp)
                        )
                        config_tmp[c_j] -= 1
                        for (target_c_old, F_config_old), V in F.items():
                            for (target_c_new, c_j_new), rij in t_updates[(target_c_old, c_j)].items():
                                F_config_new = np.array(F_config_old.array)
                                F_config_new[c_j_new] += 1
                                F_config_new = HashableArrayWrapper(F_config_new)
                                F_new[(target_c_new, F_config_new)] += V * rij
                        F = F_new

                    for (last_target_c, last_F_config), V in F.items():
                        last_F_config.array[last_target_c] += 1
                        T_new[last_F_config] += (
                            W * V
                            # * MultinomialCoefficients.coef(
                            #     tuple(config.array[config.array > 0])
                            # )
                        )
                        T_sources[last_F_config].add(config)
            all_T_sources.append(T_sources)
            T = T_new
        config = np.zeros(shape, dtype=np.uint8)
        config[-1, -1] = domain_size
        prober = all_T_sources[-1][HashableArrayWrapper(config)]
        for h in range(2, len(all_T_sources)):
            T_sources = all_T_sources[-h]
            new_prober = set()
            for config in prober:
                if config in T_sources:
                    new_prober.update(T_sources[config])
            prober = new_prober
            print(prober)
        for config, weight in T.items():
            config = config.array.sum(axis=0)
            if config.flatten()[-1] == domain_size:
                res_ += weight
        res += (res_ * graph_weight)
    return res
