"""Factorial-scaled Boundary-Profile dynamic-programming kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

from wfomc.arithmetic import ArithmeticValue

from .input import BoundaryProfileComponent
from .plan import BPNodeKind, BoundaryProfileNode, BoundaryProfilePlan

State = tuple[int, ...]


@dataclass
class BoundaryProfileRunStats:
    """Mutable execution counters reported by the native solver."""

    evaluated_nodes: int = 0
    materialized_states: int = 0
    join_state_pairs: int = 0
    zero_kernel_pairs: int = 0
    independent_root_states: int = 0


def evaluate_boundary_profile(
    component: BoundaryProfileComponent,
    plan: BoundaryProfilePlan,
    domain_size: int,
    arithmetic,
) -> tuple[ArithmeticValue, BoundaryProfileRunStats]:
    """Evaluate one planned master-sum component exactly."""

    if plan.root < 0:
        value = arithmetic.one() if domain_size == 0 else arithmetic.zero()
        return value, BoundaryProfileRunStats()
    evaluator = _Evaluator(component, plan, domain_size, arithmetic)
    message = evaluator.evaluate(plan.root)
    value = message.get((domain_size,))
    return value, evaluator.stats


class _Message:
    """Boundary table grouped by total cardinality.

    One-dimensional messages use a dense vector, which covers leaves,
    symmetric blocks, and the root without per-state dictionary overhead.
    Wider messages retain sparse dictionaries so zero constraints can prune
    large parts of the weak-composition space.
    """

    def __init__(self, dimension: int, domain_size: int, arithmetic):
        if dimension <= 0:
            raise ValueError("Boundary messages require at least one coordinate")
        self.dimension = dimension
        self.domain_size = domain_size
        self.arithmetic = arithmetic
        self._dense = (
            [arithmetic.zero() for _ in range(domain_size + 1)]
            if dimension == 1
            else None
        )
        self._buckets = (
            [dict() for _ in range(domain_size + 1)]
            if dimension > 1
            else None
        )

    def set(self, state: State, value: ArithmeticValue) -> None:
        total = sum(state)
        if self._dense is not None:
            self._dense[total] = value
            return
        assert self._buckets is not None
        if not self.arithmetic.is_zero(value):
            self._buckets[total][state] = value

    def add(self, state: State, value: ArithmeticValue) -> None:
        if self.arithmetic.is_zero(value):
            return
        total = sum(state)
        if self._dense is not None:
            self._dense[total] = self.arithmetic.add(self._dense[total], value)
            return
        assert self._buckets is not None
        bucket = self._buckets[total]
        bucket[state] = self.arithmetic.add(
            bucket.get(state, self.arithmetic.zero()),
            value,
        )

    def add_product(
        self,
        state: State,
        left: ArithmeticValue,
        right: ArithmeticValue,
    ) -> None:
        if self.arithmetic.is_zero(left) or self.arithmetic.is_zero(right):
            return
        total = sum(state)
        if self._dense is not None:
            self._dense[total] = self.arithmetic.add_product(
                self._dense[total],
                left,
                right,
            )
            return
        assert self._buckets is not None
        bucket = self._buckets[total]
        bucket[state] = self.arithmetic.add_product(
            bucket.get(state, self.arithmetic.zero()),
            left,
            right,
        )

    def entries(self, total: int) -> Iterator[tuple[State, ArithmeticValue]]:
        if self._dense is not None:
            value = self._dense[total]
            if not self.arithmetic.is_zero(value):
                yield (total,), value
            return
        assert self._buckets is not None
        yield from self._buckets[total].items()

    def get(self, state: State) -> ArithmeticValue:
        total = sum(state)
        if self._dense is not None:
            return self._dense[total]
        assert self._buckets is not None
        return self._buckets[total].get(state, self.arithmetic.zero())

    def nonzero_count(self) -> int:
        if self._dense is not None:
            return sum(
                1 for value in self._dense if not self.arithmetic.is_zero(value)
            )
        assert self._buckets is not None
        return sum(len(bucket) for bucket in self._buckets)

    def dense_values(self) -> list[ArithmeticValue]:
        if self._dense is None:
            raise ValueError("Only one-dimensional messages have dense values")
        return self._dense


class _Combinatorics:
    def __init__(self, domain_size: int, arithmetic):
        rows: list[tuple[int, ...]] = []
        current = [1]
        for _ in range(domain_size + 1):
            rows.append(tuple(current))
            current = [1] + [
                current[index] + current[index + 1]
                for index in range(len(current) - 1)
            ] + [1]
        self.rows = tuple(rows)
        self.arithmetic = arithmetic
        self._values: dict[tuple[int, int], ArithmeticValue] = {}

    def choose(self, total: int, selected: int) -> int:
        return self.rows[total][selected]

    def choose_value(self, total: int, selected: int) -> ArithmeticValue:
        key = (total, selected)
        value = self._values.get(key)
        if value is None:
            value = self.arithmetic.from_int(self.choose(total, selected))
            self._values[key] = value
        return value

    def multinomial(self, counts: Sequence[int]) -> int:
        remaining = sum(counts)
        result = 1
        for count in counts[:-1]:
            result *= self.choose(remaining, count)
            remaining -= count
        return result


class _PowerRow:
    """Incrementally materialized ``1, base, base^2, ...`` row."""

    def __init__(self, base: ArithmeticValue, arithmetic):
        self.base = base
        self.arithmetic = arithmetic
        self.values = [arithmetic.one()]

    def get(self, exponent: int) -> ArithmeticValue:
        while len(self.values) <= exponent:
            self.values.append(
                self.arithmetic.multiply(self.values[-1], self.base)
            )
        return self.values[exponent]


class _ExponentCache:
    """Bounded cache for irregular exponents such as ``c * (n-c)``."""

    _SYMBOLIC_CACHE_LIMIT = 8192

    def __init__(self, base: ArithmeticValue, max_exponent: int, arithmetic):
        self.base = base
        self.arithmetic = arithmetic
        self.sequential = not arithmetic.symbolic_variables or max_exponent <= 4096
        self.row = _PowerRow(base, arithmetic) if self.sequential else None
        self.values: dict[int, ArithmeticValue] = {0: arithmetic.one()}

    def get(self, exponent: int) -> ArithmeticValue:
        if self.row is not None:
            return self.row.get(exponent)
        cached = self.values.get(exponent)
        if cached is not None:
            return cached
        value = self.arithmetic.power(self.base, exponent)
        if len(self.values) < self._SYMBOLIC_CACHE_LIMIT:
            self.values[exponent] = value
        return value


class _Evaluator:
    def __init__(
        self,
        component: BoundaryProfileComponent,
        plan: BoundaryProfilePlan,
        domain_size: int,
        arithmetic,
    ):
        self.component = component
        self.plan = plan
        self.domain_size = domain_size
        self.arithmetic = arithmetic
        self.combinatorics = _Combinatorics(domain_size, arithmetic)
        self.stats = BoundaryProfileRunStats()

    def evaluate(self, node_index: int) -> _Message:
        node = self.plan.nodes[node_index]
        self.stats.evaluated_nodes += 1
        if node.kind is BPNodeKind.LEAF:
            message = self._leaf_message(node)
        elif node.kind is BPNodeKind.INDEPENDENT:
            message = self._independent_message(node)
        elif node.kind is BPNodeKind.SYMMETRIC:
            message = self._symmetric_message(node)
        else:
            message = self._join_message(node_index, node)
        self.stats.materialized_states += message.nonzero_count()
        return message

    def _leaf_message(self, node: BoundaryProfileNode) -> _Message:
        if len(node.members) != 1 or len(node.classes) != 1:
            raise AssertionError("A BP leaf must contain one cell and one class")
        message = _Message(1, self.domain_size, self.arithmetic)
        table = self.component.w_tables[node.members[0]]
        for count in range(self.domain_size + 1):
            message.set((count,), table[count])
        return message

    def _independent_message(self, node: BoundaryProfileNode) -> _Message:
        dimension = len(node.classes)
        message = _Message(dimension, self.domain_size, self.arithmetic)
        activities = []
        for boundary_class in node.classes:
            activity = self.arithmetic.zero()
            for cell in boundary_class:
                theta = (
                    self.component.w_tables[cell][1]
                    if self.domain_size >= 1
                    else self.arithmetic.one()
                )
                activity = self.arithmetic.add(activity, theta)
            activities.append(activity)
        powers = [_PowerRow(activity, self.arithmetic) for activity in activities]
        for total in range(self.domain_size + 1):
            for state in _compositions(dimension, total):
                value = self.arithmetic.from_int(
                    self.combinatorics.multinomial(state)
                )
                for index, count in enumerate(state):
                    if count:
                        value = self.arithmetic.multiply(value, powers[index].get(count))
                        if self.arithmetic.is_zero(value):
                            break
                message.set(state, value)
        return message

    def _symmetric_message(self, node: BoundaryProfileNode) -> _Message:
        if len(node.classes) != 1 or len(node.members) <= 1:
            raise AssertionError("A symmetric block must have one BP class and multiple cells")
        base = list(self.component.w_tables[node.members[0]][: self.domain_size + 1])
        interaction = self.component.r_matrix[node.members[0]][node.members[1]]
        max_cross = (self.domain_size // 2) * (
            self.domain_size - self.domain_size // 2
        )
        cross_powers = _ExponentCache(interaction, max_cross, self.arithmetic)

        result: list[ArithmeticValue] | None = None
        current = base
        multiplicity = len(node.members)
        while multiplicity:
            if multiplicity & 1:
                result = (
                    list(current)
                    if result is None
                    else self._combine_one_dimensional(
                        result,
                        current,
                        cross_powers,
                    )
                )
            multiplicity >>= 1
            if multiplicity:
                current = self._combine_one_dimensional(
                    current,
                    current,
                    cross_powers,
                )
        assert result is not None
        message = _Message(1, self.domain_size, self.arithmetic)
        for total, value in enumerate(result):
            message.set((total,), value)
        return message

    def _combine_one_dimensional(
        self,
        left: Sequence[ArithmeticValue],
        right: Sequence[ArithmeticValue],
        cross_powers: _ExponentCache,
    ) -> list[ArithmeticValue]:
        output = [self.arithmetic.zero() for _ in range(self.domain_size + 1)]
        for total in range(self.domain_size + 1):
            accumulator = self.arithmetic.zero()
            for left_count in range(total + 1):
                left_value = left[left_count]
                right_value = right[total - left_count]
                if self.arithmetic.is_zero(left_value) or self.arithmetic.is_zero(
                    right_value
                ):
                    continue
                factor = self.combinatorics.choose_value(total, left_count)
                cross_exponent = left_count * (total - left_count)
                if cross_exponent:
                    factor = self.arithmetic.multiply(
                        factor,
                        cross_powers.get(cross_exponent),
                    )
                factor = self.arithmetic.multiply(factor, left_value)
                accumulator = self.arithmetic.add_product(
                    accumulator,
                    factor,
                    right_value,
                )
            output[total] = accumulator
        return output

    def _join_message(
        self,
        node_index: int,
        node: BoundaryProfileNode,
    ) -> _Message:
        assert node.left is not None and node.right is not None
        left_node = self.plan.nodes[node.left]
        right_node = self.plan.nodes[node.right]
        if node_index == self.plan.root and (
            left_node.kind is BPNodeKind.INDEPENDENT
            or right_node.kind is BPNodeKind.INDEPENDENT
        ):
            if left_node.kind is BPNodeKind.INDEPENDENT:
                return self._independent_root_join(left_node, node.right)
            return self._independent_root_join(right_node, node.left)

        left_message = self.evaluate(node.left)
        right_message = self.evaluate(node.right)
        return self._ordinary_join(node, left_message, right_message)

    def _ordinary_join(
        self,
        node: BoundaryProfileNode,
        left_message: _Message,
        right_message: _Message,
    ) -> _Message:
        parent_dimension = len(node.classes)
        output = _Message(parent_dimension, self.domain_size, self.arithmetic)
        left_projection_cache: dict[State, State] = {}
        right_projection_cache: dict[State, State] = {}
        cross_rows = [
            (left_index, right_index, _PowerRow(weight, self.arithmetic))
            for left_index, right_index, weight in node.cross_nonunit
        ]

        for left_total in range(self.domain_size + 1):
            for left_state, left_value in left_message.entries(left_total):
                projected_left = left_projection_cache.get(left_state)
                if projected_left is None:
                    projected_left = _project_state(
                        left_state,
                        node.left_projection,
                        parent_dimension,
                    )
                    left_projection_cache[left_state] = projected_left

                interaction_steps = [self.arithmetic.one()] * right_message.dimension
                for left_index, right_index, powers in cross_rows:
                    count = left_state[left_index]
                    if count:
                        interaction_steps[right_index] = self.arithmetic.multiply(
                            interaction_steps[right_index],
                            powers.get(count),
                        )
                right_power_rows = [
                    None
                    if self.arithmetic.is_one(step)
                    else _PowerRow(step, self.arithmetic)
                    for step in interaction_steps
                ]

                for right_total in range(self.domain_size - left_total + 1):
                    coefficient = self.combinatorics.choose_value(
                        left_total + right_total,
                        left_total,
                    )
                    for right_state, right_value in right_message.entries(right_total):
                        self.stats.join_state_pairs += 1
                        kernel = self.arithmetic.one()
                        for right_index, powers in enumerate(right_power_rows):
                            if powers is not None and right_state[right_index]:
                                kernel = self.arithmetic.multiply(
                                    kernel,
                                    powers.get(right_state[right_index]),
                                )
                                if self.arithmetic.is_zero(kernel):
                                    break
                        if self.arithmetic.is_zero(kernel):
                            self.stats.zero_kernel_pairs += 1
                            continue
                        projected_right = right_projection_cache.get(right_state)
                        if projected_right is None:
                            projected_right = _project_state(
                                right_state,
                                node.right_projection,
                                parent_dimension,
                            )
                            right_projection_cache[right_state] = projected_right
                        parent_state = tuple(
                            left_count + right_count
                            for left_count, right_count in zip(
                                projected_left,
                                projected_right,
                            )
                        )
                        factor = self.arithmetic.multiply(coefficient, kernel)
                        factor = self.arithmetic.multiply(factor, left_value)
                        output.add_product(parent_state, factor, right_value)
        return output

    def _independent_root_join(
        self,
        independent_node: BoundaryProfileNode,
        ordinary_index: int,
    ) -> _Message:
        ordinary_node = self.plan.nodes[ordinary_index]
        ordinary_message = self.evaluate(ordinary_index)
        if len(self.plan.nodes[self.plan.root].classes) != 1:
            raise AssertionError("The full Boundary-Profile root must have one class")

        representatives = tuple(boundary_class[0] for boundary_class in ordinary_node.classes)
        interaction_rows: list[tuple[ArithmeticValue, list[_PowerRow | None]]] = []
        for cell in independent_node.members:
            theta = (
                self.component.w_tables[cell][1]
                if self.domain_size >= 1
                else self.arithmetic.one()
            )
            rows = []
            for representative in representatives:
                interaction = self.component.r_matrix[cell][representative]
                rows.append(
                    None
                    if self.arithmetic.is_one(interaction)
                    else _PowerRow(interaction, self.arithmetic)
                )
            interaction_rows.append((theta, rows))

        result = self.arithmetic.zero()
        for ordinary_total in range(self.domain_size + 1):
            coefficient = self.combinatorics.choose_value(
                self.domain_size,
                ordinary_total,
            )
            remaining = self.domain_size - ordinary_total
            for state, ordinary_value in ordinary_message.entries(ordinary_total):
                self.stats.independent_root_states += 1
                effective_sum = self.arithmetic.zero()
                for theta, rows in interaction_rows:
                    activity = theta
                    for class_index, powers in enumerate(rows):
                        if powers is not None and state[class_index]:
                            activity = self.arithmetic.multiply(
                                activity,
                                powers.get(state[class_index]),
                            )
                            if self.arithmetic.is_zero(activity):
                                break
                    effective_sum = self.arithmetic.add(effective_sum, activity)
                independent_value = self.arithmetic.power(effective_sum, remaining)
                factor = self.arithmetic.multiply(coefficient, independent_value)
                result = self.arithmetic.add_product(
                    result,
                    factor,
                    ordinary_value,
                )

        output = _Message(1, self.domain_size, self.arithmetic)
        output.set((self.domain_size,), result)
        return output


def _project_state(
    state: State,
    projection: Sequence[int],
    parent_dimension: int,
) -> State:
    result = [0] * parent_dimension
    for child_index, count in enumerate(state):
        result[projection[child_index]] += count
    return tuple(result)


def _compositions(length: int, total: int) -> Iterator[State]:
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for suffix in _compositions(length - 1, total - first):
            yield (first,) + suffix


__all__ = [
    "BoundaryProfileRunStats",
    "evaluate_boundary_profile",
]
