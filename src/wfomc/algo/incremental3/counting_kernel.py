"""Incremental WFOMC3 counting-DP kernel (framework-owned).

Pure dynamic-programming core for the incremental3 algorithm: configuration
space, weight/transition tables, and the domain recursion. Operates on plain
cells, weight tables, and a counting-state value, so it is independent of any
WFOMC context.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import product
from typing import Any, Callable, TypeAlias

from wfomc.arithmetic import ArithmeticValue

State: TypeAlias = tuple[int, ...]
Config: TypeAlias = tuple[int, ...]
CountingState: TypeAlias = Any


class ConfigSpace:
    """Compact immutable representation for DP configurations."""

    __slots__ = (
        "shape",
        "zero",
        "offset_to_state",
        "state_to_offset",
        "_nonzero_cache",
    )

    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(shape)
        ranges = [range(dim) for dim in self.shape]
        self.offset_to_state: tuple[State, ...] = tuple(product(*ranges))
        self.state_to_offset: dict[State, int] = {
            state: offset for offset, state in enumerate(self.offset_to_state)
        }
        self.zero: Config = (0,) * len(self.offset_to_state)
        self._nonzero_cache: dict[Config, tuple[State, ...]] = {}

    def offset(self, state: State) -> int:
        return self.state_to_offset[state]

    def count(self, config: Config, state: State) -> int:
        return config[self.offset(state)]

    def inc(self, config: Config, state: State, amount: int = 1) -> Config:
        offset = self.offset(state)
        return config[:offset] + (config[offset] + amount,) + config[offset + 1 :]

    def dec(self, config: Config, state: State, amount: int = 1) -> Config:
        offset = self.offset(state)
        return config[:offset] + (config[offset] - amount,) + config[offset + 1 :]

    @staticmethod
    def add(left: Config, right: Config) -> Config:
        return tuple(a + b for a, b in zip(left, right))

    def nonzero_states(self, config: Config) -> tuple[State, ...]:
        cached = self._nonzero_cache.get(config)
        if cached is None:
            cached = tuple(
                self.offset_to_state[offset]
                for offset, count in enumerate(config)
                if count > 0
            )
            self._nonzero_cache[config] = cached
        return cached


class ConfigUpdater:
    """
    Memoised configuration updater for efficient state transitions.

    _cache structure: {(target_c, other_c): {j: H_dict}}
    where H_dict maps (target_c_new, H_config_new) to a branch-domain weight,
    recording the cumulative weight of pairing target_c with j other_c elements.
    """

    def __init__(self, t_update_dict, space: ConfigSpace, arithmetic):
        self.t_update_dict = t_update_dict
        self.space = space
        self.arithmetic = arithmetic
        self._cache: dict[tuple[State, State], dict[int, dict]] = {}

    def f(self, target_c: State, other_c: State, other_count: int):
        """Return the weighted outcome of pairing target_c with other elements."""
        key = (target_c, other_c)
        sub = self._cache.get(key)
        if sub is None:
            sub = {}
            self._cache[key] = sub
            num_start = 0
        else:
            num_start = other_count
            while num_start not in sub and num_start > 0:
                num_start -= 1

        if num_start == 0:
            H = {(target_c, self.space.zero): self.arithmetic.one()}
        else:
            H = sub[num_start]

        for j in range(num_start + 1, other_count + 1):
            H_new = defaultdict(self.arithmetic.zero)
            for (tc_old, hc_old), W in H.items():
                for (tc_new, oc_new), rij in self.t_update_dict[
                    (tc_old, other_c)
                ].items():
                    hc_new = self.space.inc(hc_old, oc_new)
                    H_new[(tc_new, hc_new)] += W * rij
            H = H_new
            sub[j] = H

        return H


def build_t_update_dict(
    r,
    n_cells: int,
    state: CountingState,
    arithmetic,
) -> defaultdict:
    """Build the state transition lookup table for all cell-pair combinations."""
    t_update_dict = defaultdict(lambda: defaultdict(arithmetic.zero))

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
        all_ts = list(
            product(
                *(
                    [tuple(range(2)) for _ in state.ext_preds]
                    + [tuple(range(k + 1)) for k in state.cnt_params]
                )
            )
        )

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


def _stop_condition(target_c: State, state: CountingState) -> bool:
    """Check whether the target element's state satisfies all counting constraints."""
    pred_state = target_c[1:]
    if state.exist_le:
        for i in range(len(pred_state)):
            if i not in state.le_index and pred_state[i] != 0:
                return False
        return True
    else:
        return all(s == 0 for s in pred_state)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


def _make_domain_recursion(
    t_update_dict,
    space: ConfigSpace,
    cs: CountingState,
    has_linear_order: bool,
    arithmetic,
) -> Callable[[Config], ArithmeticValue]:
    """Return a memoised domain_recursion function scoped to one cell graph."""
    updater = ConfigUpdater(t_update_dict, space, arithmetic)
    f = updater.f
    cache: dict[Config, ArithmeticValue] = {}

    def domain_recursion(config: Config):
        if config in cache:
            return cache[config]

        if sum(config) == 0:
            return arithmetic.one()

        result = arithmetic.zero()
        nonzero_states = space.nonzero_states(config)

        if has_linear_order:
            target_c_list = nonzero_states
        else:
            target_c_list = (nonzero_states[-1],)

        for target_c in target_c_list:
            T = defaultdict(arithmetic.zero)
            config_new = space.dec(config, target_c)

            G = {(target_c, space.zero): arithmetic.one()}

            for other_c in space.nonzero_states(config_new):
                G_new = defaultdict(arithmetic.zero)
                other_count = space.count(config_new, other_c)

                for (tc, G_config), W in G.items():
                    for (tc_new, H_config_new), weight_H in f(
                        tc,
                        other_c,
                        other_count,
                    ).items():
                        G_config_new = space.add(G_config, H_config_new)

                        if has_linear_order:
                            denom = 1
                            for count in H_config_new:
                                if count > 1:
                                    denom *= math.factorial(count)
                            weight_H = weight_H * arithmetic.from_fraction(
                                1,
                                math.factorial(other_count) // denom,
                            )

                        G_new[(tc_new, G_config_new)] += W * weight_H
                G = G_new

            for (target_c, G_config), W in G.items():
                if _stop_condition(target_c, cs):
                    T[G_config] += W

            result_of_target_c = arithmetic.zero()
            for T_config, weight in T.items():
                result_of_target_c += weight * domain_recursion(T_config)
            result += result_of_target_c

        cache[config] = result
        return result

    return domain_recursion


__all__ = [
    "ConfigSpace",
    "ConfigUpdater",
    "build_t_update_dict",
    "_make_domain_recursion",
]
