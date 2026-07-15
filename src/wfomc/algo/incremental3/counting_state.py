"""Counting automata materialized for the incremental3 algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wfomc.fol import Predicate
from wfomc.fol.normal_form import (
    C2NormalForm,
    CountSection,
    ForallCountSection,
)

if TYPE_CHECKING:
    from wfomc.cell_graph import Cell


@dataclass(frozen=True)
class RowCounterSpec:
    """Finite counter for one binary counting predicate.

    ``true_transitions[state]`` is the next state after observing one true
    binary tuple. ``None`` rejects an overflow immediately. A marker predicate
    turns the direct constraint into an equivalence between marker polarity
    and membership in ``accepting_states``.
    """

    predicate: Predicate
    true_transitions: tuple[int | None, ...]
    accepting_states: frozenset[int]
    marker_predicate: Predicate | None = None

    @property
    def state_count(self) -> int:
        return len(self.true_transitions)

    def initial_state(self, diagonal_is_true: bool) -> int | None:
        return self.true_transitions[0] if diagonal_is_true else 0

    def accepting_states_for_cell(self, cell: "Cell") -> frozenset[int]:
        if self.marker_predicate is None or cell.is_positive(self.marker_predicate):
            return self.accepting_states
        return frozenset(range(self.state_count)) - self.accepting_states


@dataclass(frozen=True)
class CountingState:
    row_counters: tuple[RowCounterSpec, ...]

    @property
    def counter_predicates(self) -> tuple[Predicate, ...]:
        return tuple(counter.predicate for counter in self.row_counters)

    @property
    def projected_predicates(self) -> tuple[Predicate, ...]:
        unique: list[Predicate] = []
        for predicate in self.counter_predicates:
            if predicate not in unique:
                unique.append(predicate)
        return tuple(unique)

    @property
    def counter_projection_indices(self) -> tuple[int, ...]:
        projected = self.projected_predicates
        return tuple(
            projected.index(predicate) for predicate in self.counter_predicates
        )

    @property
    def c_type_shape(self) -> tuple[int, ...]:
        return tuple(counter.state_count for counter in self.row_counters)


@dataclass(frozen=True)
class GlobalCountSpec:
    predicate: Predicate
    comparator: str
    count: int | tuple[int, int]
    marker_predicate: Predicate | None = None


class UnaryCardinalityMasks:
    """Global unary counts evaluated against a materialized cell allocation."""

    def __init__(self) -> None:
        self.constraints: list[GlobalCountSpec] = []

    def add(
        self,
        section: CountSection,
        *,
        marker_predicate: Predicate | None = None,
    ) -> None:
        self.constraints.append(
            GlobalCountSpec(
                predicate=_counting_predicate("unary", section.body),
                comparator=section.comparator,
                count=section.count,
                marker_predicate=marker_predicate,
            )
        )

    def required_predicates(self) -> frozenset[Predicate]:
        return frozenset(spec.predicate for spec in self.constraints)

    def build_mask(
        self,
        cells: tuple["Cell", ...],
        nullary_assignments: tuple[tuple[Predicate, bool], ...] = (),
    ) -> tuple[tuple[np.ndarray, GlobalCountSpec, bool | None], ...]:
        assignment = {
            predicate.cache_key_parts(): value
            for predicate, value in nullary_assignments
        }
        masks = []
        for spec in self.constraints:
            marker_value = None
            if spec.marker_predicate is not None:
                key = spec.marker_predicate.cache_key_parts()
                if key not in assignment:
                    raise RuntimeError(
                        f"Missing nullary count marker assignment for "
                        f"{spec.marker_predicate}"
                    )
                marker_value = assignment[key]
            masks.append(
                (
                    np.fromiter(
                        (
                            1 if cell.is_positive(spec.predicate) else 0
                            for cell in cells
                        ),
                        dtype=np.int8,
                        count=len(cells),
                    ),
                    spec,
                    marker_value,
                )
            )
        return tuple(masks)

    def check(
        self,
        config: tuple[int, ...],
        masks: tuple[tuple[np.ndarray, GlobalCountSpec, bool | None], ...],
    ) -> bool:
        if not masks:
            return False
        vector = np.fromiter(config, dtype=np.int32)
        for mask, spec, marker_value in masks:
            actual = int(mask @ vector)
            holds = _count_holds(spec.comparator, spec.count, actual)
            if holds != (True if marker_value is None else marker_value):
                return True
        return False


def build_counting_state_for_normal_form(
    normal_form: C2NormalForm,
    *,
    domain_size: int | None = None,
) -> tuple[CountingState, UnaryCardinalityMasks]:
    global_counts = UnaryCardinalityMasks()
    row_counters = [
        _row_counter(section, domain_size=domain_size)
        for section in normal_form.forall_counts
    ]
    for section in normal_form.counts:
        global_counts.add(section)

    for definition in normal_form.count_definitions:
        section = definition.section
        marker = definition.marker.predicate
        if isinstance(section, CountSection):
            global_counts.add(section, marker_predicate=marker)
        elif isinstance(section, ForallCountSection):
            row_counters.append(
                _row_counter(
                    section,
                    marker_predicate=marker,
                    domain_size=domain_size,
                )
            )
        else:
            raise TypeError(
                f"Unsupported count definition section: {type(section).__name__}"
            )

    return CountingState(tuple(row_counters)), global_counts


def _row_counter(
    section: ForallCountSection,
    *,
    marker_predicate: Predicate | None = None,
    domain_size: int | None = None,
) -> RowCounterSpec:
    predicate = _counting_predicate("binary", section.body)
    comparator = section.comparator
    count = section.count

    if comparator == "mod":
        remainder, modulus = count
        remainder = int(remainder)
        modulus = int(modulus)
        if domain_size is None or modulus <= domain_size:
            return RowCounterSpec(
                predicate=predicate,
                true_transitions=tuple(
                    (state + 1) % modulus for state in range(modulus)
                ),
                accepting_states=frozenset({remainder}),
                marker_predicate=marker_predicate,
            )
        if remainder > domain_size:
            return _constant_row_counter(predicate, False, marker_predicate)
        mode, threshold = "eq", remainder
    else:
        mode, threshold = _canonical_comparator(comparator, int(count))

    if mode == "false":
        return _constant_row_counter(predicate, False, marker_predicate)
    constant = _constant_row_condition(mode, threshold, domain_size)
    if constant is not None:
        return _constant_row_counter(predicate, constant, marker_predicate)
    if mode == "eq":
        keep_overflow = marker_predicate is not None
        transitions = _upper_bounded_transitions(threshold, keep_overflow)
        accepting = frozenset({threshold})
    elif mode == "le":
        keep_overflow = marker_predicate is not None
        transitions = _upper_bounded_transitions(threshold, keep_overflow)
        accepting = frozenset(range(threshold + 1))
    elif mode == "ge":
        transitions = tuple(
            min(state + 1, threshold) for state in range(threshold + 1)
        )
        accepting = frozenset({threshold})
    elif mode == "ne":
        transitions = tuple(
            min(state + 1, threshold + 1) for state in range(threshold + 2)
        )
        accepting = frozenset(range(threshold + 2)) - {threshold}
    else:
        raise AssertionError(f"Unknown canonical row count mode: {mode}")

    return RowCounterSpec(
        predicate=predicate,
        true_transitions=transitions,
        accepting_states=accepting,
        marker_predicate=marker_predicate,
    )


def _constant_row_condition(
    mode: str,
    threshold: int,
    domain_size: int | None,
) -> bool | None:
    if domain_size is None:
        return None
    if mode == "eq" and threshold > domain_size:
        return False
    if mode == "ne" and threshold > domain_size:
        return True
    if mode == "le" and threshold >= domain_size:
        return True
    if mode == "ge":
        if threshold == 0:
            return True
        if threshold > domain_size:
            return False
    return None


def _constant_row_counter(
    predicate: Predicate,
    value: bool,
    marker_predicate: Predicate | None,
) -> RowCounterSpec:
    return RowCounterSpec(
        predicate=predicate,
        true_transitions=(0,),
        accepting_states=frozenset({0}) if value else frozenset(),
        marker_predicate=marker_predicate,
    )


def _upper_bounded_transitions(
    threshold: int,
    keep_overflow: bool,
) -> tuple[int | None, ...]:
    if keep_overflow:
        overflow = threshold + 1
        return tuple(min(state + 1, overflow) for state in range(threshold + 2))
    return tuple(
        state + 1 if state < threshold else None
        for state in range(threshold + 1)
    )


def _canonical_comparator(comparator: str, count: int) -> tuple[str, int]:
    if comparator == "=":
        return "eq", count
    if comparator == "!=":
        return "ne", count
    if comparator == "<=":
        return "le", count
    if comparator == "<":
        return ("false", 0) if count == 0 else ("le", count - 1)
    if comparator == ">=":
        return "ge", count
    if comparator == ">":
        return "ge", count + 1
    raise ValueError(f"Unsupported counting comparator: {comparator!r}")


def _count_holds(
    comparator: str,
    count: int | tuple[int, int],
    actual: int,
) -> bool:
    if comparator == "mod":
        remainder, modulus = count
        return actual % int(modulus) == int(remainder)
    threshold = int(count)
    if comparator == "=":
        return actual == threshold
    if comparator == "!=":
        return actual != threshold
    if comparator == "<=":
        return actual <= threshold
    if comparator == "<":
        return actual < threshold
    if comparator == ">=":
        return actual >= threshold
    if comparator == ">":
        return actual > threshold
    raise ValueError(f"Unsupported counting comparator: {comparator!r}")


def _counting_predicate(kind: str, body: object) -> Predicate:
    from wfomc.fol import Atom

    expected_arity = 1 if kind == "unary" else 2
    if not isinstance(body, Atom) or len(body.terms) != expected_arity:
        raise TypeError(
            f"{kind} counting section requires an atomic body with arity "
            f"{expected_arity}, got {body}"
        )
    return body.predicate


__all__ = [
    "CountingState",
    "GlobalCountSpec",
    "RowCounterSpec",
    "UnaryCardinalityMasks",
    "build_counting_state_for_normal_form",
]
