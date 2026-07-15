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
from typing import TYPE_CHECKING, Callable, TypeAlias

from wfomc.arithmetic import ArithmeticValue

if TYPE_CHECKING:
    from .counting_state import CountingState

State: TypeAlias = tuple[int, ...]
Config: TypeAlias = tuple[int, ...]


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
        self.add_product = arithmetic.add_product
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
                    outcome = (tc_new, hc_new)
                    H_new[outcome] = self.add_product(
                        H_new[outcome],
                        W,
                        rij,
                    )
            H = H_new
            sub[j] = H

        return H


def _build_elimination_orders(
    t_update_dict,
    states: tuple[State, ...],
) -> tuple[tuple[State, ...], dict[State, tuple[State, ...]]]:
    """Plan deterministic fail-first orders from transition structure.

    Transition structure determines the priority; state offsets are only the
    final tie-break for structurally indistinguishable states.  A state is
    preferred when it is incompatible with more partners, then when its
    compatible transitions create fewer successor states.  For a chosen
    target, incompatible and low-fanout partners are visited first so dead
    branches and small frontiers are exposed early.
    """

    pair_keys = {
        (target, other): _transition_order_key(t_update_dict, target, other)
        for target in states
        for other in states
    }
    fingerprints = {
        target: tuple(sorted(pair_keys[(target, other)] for other in states))
        for target in states
    }

    def target_key(target: State):
        keys = tuple(pair_keys[(target, other)] for other in states)
        incompatible = sum(key[0] == 0 for key in keys)
        branching_excess = sum(max(0, key[1] - 1) for key in keys)
        successor_spread = sum(key[2] * key[3] for key in keys)
        return (
            -incompatible,
            branching_excess,
            successor_spread,
            fingerprints[target],
            target,
        )

    target_order = tuple(sorted(states, key=target_key))
    other_orders = {
        target: tuple(
            sorted(
                states,
                key=lambda other: (
                    pair_keys[(target, other)],
                    fingerprints[other],
                    other,
                ),
            )
        )
        for target in states
    }
    return target_order, other_orders


def _transition_order_key(t_update_dict, target: State, other: State):
    outcomes = t_update_dict.get((target, other))
    if not outcomes:
        return (0, 0, 0, 0)
    target_successors = {outcome[0] for outcome in outcomes}
    other_successors = {outcome[1] for outcome in outcomes}
    return (
        1,
        len(outcomes),
        len(target_successors),
        len(other_successors),
    )


def build_t_update_dict(
    r,
    n_cells: int,
    state: CountingState,
    arithmetic,
) -> defaultdict:
    """Build the state transition lookup table for all cell-pair combinations."""
    t_update_dict = defaultdict(lambda: defaultdict(arithmetic.zero))
    all_ts = list(
        product(*(range(counter.state_count) for counter in state.row_counters))
    )

    def advance(counter_state, delta):
        updated = []
        for value, observed, counter in zip(
            counter_state,
            delta,
            state.row_counters,
        ):
            next_value = counter.true_transitions[value] if observed else value
            if next_value is None:
                return None
            updated.append(next_value)
        return tuple(updated)

    for i in range(n_cells):
        for j in range(n_cells):
            for t1 in all_ts:
                for t2 in all_ts:
                    for (dt, reverse_dt), rijt in r[(i, j)].items():
                        t1_new = advance(t1, dt)
                        t2_new = advance(t2, reverse_dt)
                        if t1_new is None or t2_new is None:
                            continue

                        c1 = (i,) + t1
                        c2 = (j,) + t2
                        transition = (
                            (i,) + t1_new,
                            (j,) + t2_new,
                        )
                        t_update_dict[(c1, c2)][transition] = arithmetic.add(
                            t_update_dict[(c1, c2)][transition],
                            rijt,
                        )

    return t_update_dict


def _stop_condition(
    target_c: State,
    accepting_states: tuple[tuple[frozenset[int], ...], ...],
) -> bool:
    """Check whether the target element's state satisfies all counting constraints."""
    cell_index = target_c[0]
    return all(
        value in accepted
        for value, accepted in zip(
            target_c[1:],
            accepting_states[cell_index],
        )
    )


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


def _make_domain_recursion(
    t_update_dict,
    space: ConfigSpace,
    accepting_states: tuple[tuple[frozenset[int], ...], ...],
    has_linear_order: bool,
    arithmetic,
) -> Callable[[Config], ArithmeticValue]:
    """Return a memoised domain_recursion function scoped to one cell graph."""
    updater = ConfigUpdater(t_update_dict, space, arithmetic)
    f = updater.f
    add_product = arithmetic.add_product
    cache: dict[Config, ArithmeticValue] = {}
    target_order, other_orders = _build_elimination_orders(
        t_update_dict,
        space.offset_to_state,
    )
    target_rank = {state: rank for rank, state in enumerate(target_order)}
    other_ranks = {
        target: {state: rank for rank, state in enumerate(order)}
        for target, order in other_orders.items()
    }

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
            target_c_list = (min(nonzero_states, key=target_rank.__getitem__),)

        for target_c in target_c_list:
            T = defaultdict(arithmetic.zero)
            config_new = space.dec(config, target_c)

            G = {(target_c, space.zero): arithmetic.one()}

            other_states = space.nonzero_states(config_new)
            for other_c in sorted(
                other_states,
                key=other_ranks[target_c].__getitem__,
            ):
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
                            weight_H = arithmetic.multiply(
                                weight_H,
                                arithmetic.from_fraction(
                                    1,
                                    math.factorial(other_count) // denom,
                                ),
                            )

                        outcome = (tc_new, G_config_new)
                        G_new[outcome] = add_product(
                            G_new[outcome],
                            W,
                            weight_H,
                        )
                G = G_new
                if not G:
                    break

            for (target_c, G_config), W in G.items():
                if _stop_condition(target_c, accepting_states):
                    T[G_config] = arithmetic.add(T[G_config], W)

            result_of_target_c = arithmetic.zero()
            for T_config, weight in T.items():
                result_of_target_c = add_product(
                    result_of_target_c,
                    weight,
                    domain_recursion(T_config),
                )
            result = arithmetic.add(result, result_of_target_c)

        cache[config] = result
        return result

    return domain_recursion


__all__ = [
    "ConfigSpace",
    "ConfigUpdater",
    "build_t_update_dict",
    "_build_elimination_orders",
    "_make_domain_recursion",
]
