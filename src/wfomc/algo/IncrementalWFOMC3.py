from __future__ import annotations
import hashlib
import math
from collections import defaultdict
from itertools import product
from typing import Callable

import numpy as np
from loguru import logger

from wfomc.cell_graph import build_cell_graphs
from wfomc.context import IncrementalWFOMC3Context, CountingState
from wfomc.fol import Const, Pred
from wfomc.utils import multinomial, MultinomialCoefficients, Rational, expand, RingElement


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

class HashableArrayWrapper:
    """Wraps a NumPy array to make it hashable (for use as a dict key)."""

    def __init__(self, input_array: np.ndarray):
        self.array = input_array.astype(np.uint8, copy=False)

    def __hash__(self):
        return int(hashlib.sha1(self.array).hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return int(hashlib.sha1(self.array).hexdigest(), 16) == int(
                hashlib.sha1(other.array).hexdigest(), 16
            )
        return False

    def __repr__(self):
        return f"HashableArrayWrapper({self.array})"


class ConfigUpdater:
    """
    Memoised configuration updater for efficient state transitions.

    _cache structure: {(target_c, other_c): {j: H_dict}}
    where H_dict maps (target_c_new, H_config_new) → Rational weight,
    recording the cumulative weight of pairing target_c with j other_c elements.
    """

    def __init__(self, t_update_dict, c1_type_shape):
        self.t_update_dict = t_update_dict
        self.c1_type_shape = c1_type_shape
        self._cache: dict = {}

    def f(self, target_c, other_c, l):
        """Return the weighted outcome of pairing target_c with l other_c elements."""
        if (target_c, other_c) in self._cache:
            sub = self._cache[(target_c, other_c)]
            num_start = l
            while num_start not in sub and num_start > 0:
                num_start -= 1
        else:
            self._cache[(target_c, other_c)] = {}
            num_start = 0

        if num_start == 0:
            H_config = HashableArrayWrapper(np.zeros(self.c1_type_shape, dtype=np.uint8))
            H = {(target_c, H_config): Rational(1, 1)}
        else:
            H = self._cache[(target_c, other_c)][num_start]

        for j in range(num_start + 1, l + 1):
            H_new = defaultdict(lambda: Rational(0, 1))
            for (tc_old, hc_old), W in H.items():
                for (tc_new, oc_new), rij in self.t_update_dict[(tc_old, other_c)].items():
                    hc_new = HashableArrayWrapper(np.array(hc_old.array))
                    hc_new.array[oc_new] += 1
                    H_new[(tc_new, hc_new)] += W * rij
            H = H_new
            self._cache[(target_c, other_c)][j] = H

        return H


# ---------------------------------------------------------------------------
# Cell-graph weight builders
# ---------------------------------------------------------------------------

def build_weight(cells, cell_graph, state: CountingState) -> tuple:
    """
    Construct the initial-state mapping w2t, cell weight dict w, and
    binary relationship dict r from cell-graph data and counting state.
    """
    n_cells = len(cells)
    w2t = {}
    w = defaultdict(lambda: Rational(0, 1))
    r = defaultdict(lambda: defaultdict(lambda: Rational(0, 1)))

    for i in range(n_cells):
        cell_weight = cell_graph.get_cell_weight(cells[i])
        logger.debug("Cell {} weight: {}", i, cell_weight)
        t = []

        for pred in state.ext_preds:
            t.append(0 if cells[i].is_positive(pred) else 1)
        logger.debug("Cell {} existential quantifier state: {}", i, t)

        for idx, (pred, param) in enumerate(zip(state.cnt_preds, state.cnt_params)):
            if cells[i].is_positive(pred):
                t.append(
                    state.cnt_remainder[idx] - 1
                    if state.exist_mod and idx in state.mod_pred_index
                    else param - 1
                )
            else:
                t.append(
                    state.cnt_remainder[idx]
                    if state.exist_mod and idx in state.mod_pred_index
                    else param
                )
        logger.debug("Cell {} counting quantifier state: {}", i, t)

        w2t[i] = tuple(t)
        w[i] = w[i] + cell_weight

        for j in range(n_cells):
            for evi_idx, evidence in enumerate(state.binary_evidence):
                two_table_weight = cell_graph.get_two_table_weight(
                    (cells[i], cells[j]), evidence
                )
                if two_table_weight == Rational(0, 1):
                    continue
                t_fwd, t_rev = [], []
                for pred_idx, pred in enumerate(state.ext_preds + state.cnt_preds):
                    t_rev.append(1 if (evi_idx >> (2 * pred_idx)) & 1 else 0)
                    t_fwd.append(1 if (evi_idx >> (2 * pred_idx + 1)) & 1 else 0)
                r[(i, j)][(tuple(t_fwd), tuple(t_rev))] = two_table_weight

    return w2t, w, r


def build_t_update_dict(r, n_cells: int, state: CountingState) -> defaultdict:
    """Build the state transition lookup table for all cell-pair combinations."""
    t_update_dict = defaultdict(lambda: defaultdict(lambda: Rational(0, 1)))

    n_ext = len(state.ext_preds)
    n_cnt = len(state.cnt_params)

    if state.exist_mod:
        ranges = [tuple(range(2)) for _ in state.ext_preds]
        for p, k in enumerate(state.cnt_params):
            ranges.append(
                tuple(range(k)) if p in state.mod_pred_index else tuple(range(k + 1))
            )
        all_ts = list(product(*ranges))
    else:
        all_ts = list(product(*(
            [tuple(range(2)) for _ in state.ext_preds]
            + [tuple(range(k + 1)) for k in state.cnt_params]
        )))

    for i in range(n_cells):
        for j in range(n_cells):
            for t1 in all_ts:
                for t2 in all_ts:
                    for (dt, reverse_dt), rijt in r[(i, j)].items():
                        t1_new = [x - y for x, y in zip(t1, dt)]
                        t2_new = [x - y for x, y in zip(t2, reverse_dt)]

                        if state.exist_mod:
                            for p, k_i in enumerate(state.cnt_params):
                                slot = n_ext + p
                                if p in state.mod_pred_index:
                                    t1_new[slot] %= k_i
                                    t2_new[slot] %= k_i

                        if any(
                            t1_new[n_ext + p] < 0 or t2_new[n_ext + p] < 0
                            for p in range(n_cnt)
                        ):
                            continue

                        for slot in range(n_ext):
                            t1_new[slot] = max(t1_new[slot], 0)
                            t2_new[slot] = max(t2_new[slot], 0)

                        c1 = (i,) + t1
                        c2 = (j,) + t2
                        t_update_dict[(c1, c2)][
                            ((i,) + tuple(t1_new), (j,) + tuple(t2_new))
                        ] += rijt

    return t_update_dict


def _stop_condition(target_c, state: CountingState):
    """Check whether the target element's state satisfies all counting constraints."""
    pred_state = target_c[1:]
    if state.exist_le:
        for i in range(len(pred_state)):
            if i not in state.le_index and pred_state[i] != 0:
                return False
    else:
        return all(s == 0 for s in pred_state)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

def _make_domain_recursion(
    t_update_dict, c1_type_shape: tuple, cs: CountingState, has_linear_order: bool
) -> Callable:
    """Return a memoised domain_recursion function scoped to one cell graph."""
    updater = ConfigUpdater(t_update_dict, c1_type_shape)
    f = updater.f
    cache: dict = {}

    def domain_recursion(config):
        if config in cache:
            return cache[config]

        if config.array.sum() == 0:
            return Rational(1, 1)

        result = Rational(0, 1)

        if has_linear_order:
            target_c_list = [tuple(i) for i in np.argwhere(config.array > 0)]
        else:
            target_c_list = [tuple(np.argwhere(config.array > 0)[-1])]

        for target_c in target_c_list:
            T = defaultdict(lambda: Rational(0, 1))
            config_new = HashableArrayWrapper(np.array(config.array, copy=True, dtype=np.uint8))
            config_new.array[target_c] -= 1

            G = {(target_c, HashableArrayWrapper(np.zeros(c1_type_shape, dtype=np.uint8))): Rational(1, 1)}

            for other_c in np.argwhere(config_new.array > 0):
                other_c = tuple(other_c.flatten())
                G_new = defaultdict(lambda: Rational(0, 1))
                l = config_new.array[other_c]

                for (tc, G_config), W in G.items():
                    for (tc_new, H_config_new), weight_H in f(tc, other_c, l).items():
                        G_config_new = HashableArrayWrapper(G_config.array + H_config_new.array)

                        if has_linear_order:
                            denom = 1
                            for count in H_config_new.array.flatten():
                                if count > 1:
                                    denom *= math.factorial(count)
                            weight_H = weight_H * Rational(1, math.factorial(l) // denom)

                        G_new[(tc_new, G_config_new)] += W * weight_H
                G = G_new

            for (target_c, G_config), W in G.items():
                if _stop_condition(target_c, cs):
                    T[G_config] += W

            result_of_target_c = Rational(0, 1)
            for T_config, weight in T.items():
                result_of_target_c += weight * domain_recursion(T_config)
            result += result_of_target_c

        cache[config] = result
        return result

    return domain_recursion


def incremental_wfomc3(context: IncrementalWFOMC3Context) -> RingElement:
    domain: set[Const] = context.domain
    formula = context.formula
    get_weight = context.get_weight
    leq_pred: Pred = context.leq_pred
    cs = context.counting_state
    has_lo = context.contain_linear_order_axiom()

    WFOMC_result = Rational(0, 1)
    domain_size = len(domain)
    MultinomialCoefficients.setup(domain_size)

    for cell_graph, graph_weight in build_cell_graphs(formula, get_weight, leq_pred):
        cells = cell_graph.get_cells()
        n_cells = len(cells)

        w2t, w, r = build_weight(cells, cell_graph, cs)
        logger.debug("Weight mapping w2t: {}", w2t)
        logger.debug("Weight w: {}", w)

        unary_mask = context.unary_handler.build_mask(cells)
        t_update_dict = build_t_update_dict(r, n_cells, cs)
        c1_type_shape = (n_cells,) + tuple(cs.c_type_shape)
        domain_recursion = _make_domain_recursion(t_update_dict, c1_type_shape, cs, has_lo)

        for config in multinomial(n_cells, domain_size):
            logger.debug("Config: {}", config)
            if any(context.unary_handler.check(config, unary_mask)):
                continue

            init_config = np.zeros(c1_type_shape, dtype=np.uint8)
            W = Rational(1, 1)
            for i, n in enumerate(config):
                init_config[(i,) + w2t[i]] = n
                W = W * (w[i] ** n)

            result_config = domain_recursion(HashableArrayWrapper(init_config))

            if has_lo:
                WFOMC_result += W * result_config * graph_weight
            else:
                WFOMC_result += MultinomialCoefficients.coef(config) * W * result_config * graph_weight

    return expand(WFOMC_result)
