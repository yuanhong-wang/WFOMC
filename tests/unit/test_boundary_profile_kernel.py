from __future__ import annotations

import math
import random

import pytest
from flint import fmpq

from wfomc.algo.boundary_profile.input import BoundaryProfileComponent
from wfomc.algo.boundary_profile.kernel import evaluate_boundary_profile
from wfomc.algo.boundary_profile.plan import BPNodeKind, build_boundary_profile_plan
from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext


def _compositions(length: int, total: int):
    if length == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for suffix in _compositions(length - 1, total - first):
            yield (first,) + suffix


def _direct_master_sum(component, domain_size, arithmetic):
    if not component.w_tables:
        return arithmetic.one() if domain_size == 0 else arithmetic.zero()
    result = arithmetic.zero()
    for counts in _compositions(len(component.w_tables), domain_size):
        coefficient = math.factorial(domain_size)
        for count in counts:
            coefficient //= math.factorial(count)
        value = arithmetic.from_int(coefficient)
        for cell, count in enumerate(counts):
            value = arithmetic.multiply(value, component.w_tables[cell][count])
        for left in range(len(counts)):
            for right in range(left + 1, len(counts)):
                value = arithmetic.multiply(
                    value,
                    arithmetic.power(
                        component.r_matrix[left][right],
                        counts[left] * counts[right],
                    ),
                )
        result = arithmetic.add(result, value)
    return result


@pytest.mark.parametrize("domain_size", range(6))
def test_kernel_matches_direct_master_sum(domain_size):
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    component = BoundaryProfileComponent(
        cell_weights=tuple(fmpq(base) for base in (2, 3, 5, 7)),
        w_tables=tuple(
            tuple(fmpq(base**count) for count in range(domain_size + 1))
            for base in (2, 3, 5, 7)
        ),
        r_matrix=tuple(
            tuple(fmpq(value) for value in row)
            for row in (
                (1, 1, 2, 3),
                (1, 1, 2, 5),
                (2, 2, 1, 7),
                (3, 5, 7, 1),
            )
        ),
        graph_weight=fmpq(1),
    )
    plan = build_boundary_profile_plan(component, domain_size, arithmetic)

    actual, _stats = evaluate_boundary_profile(
        component,
        plan,
        domain_size,
        arithmetic,
    )

    assert actual == _direct_master_sum(component, domain_size, arithmetic)


@pytest.mark.parametrize("seed", range(12))
def test_random_small_master_sums_match_direct_enumeration(seed):
    rng = random.Random(seed)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    domain_size = rng.randrange(0, 5)
    cell_count = rng.randrange(1, 6)
    bases = [rng.randrange(0, 6) for _ in range(cell_count)]
    diagonals = [rng.randrange(0, 4) for _ in range(cell_count)]
    interactions = [[1 for _ in range(cell_count)] for _ in range(cell_count)]
    for cell, diagonal in enumerate(diagonals):
        interactions[cell][cell] = diagonal
    for left in range(cell_count):
        for right in range(left + 1, cell_count):
            interactions[left][right] = interactions[right][left] = rng.randrange(0, 4)
    component = BoundaryProfileComponent(
        cell_weights=tuple(fmpq(base) for base in bases),
        w_tables=tuple(
            tuple(
                fmpq(base**count * diagonal ** (count * (count - 1) // 2))
                for count in range(domain_size + 1)
            )
            for base, diagonal in zip(bases, diagonals)
        ),
        r_matrix=tuple(
            tuple(fmpq(value) for value in row) for row in interactions
        ),
        graph_weight=fmpq(1),
    )
    plan = build_boundary_profile_plan(component, domain_size, arithmetic)

    actual, _stats = evaluate_boundary_profile(
        component,
        plan,
        domain_size,
        arithmetic,
    )

    assert actual == _direct_master_sum(component, domain_size, arithmetic)


def test_independent_root_closing_matches_multinomial_theorem():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    domain_size = 9
    one = arithmetic.one()
    component = BoundaryProfileComponent(
        cell_weights=tuple(fmpq(base) for base in (2, 3, 5)),
        w_tables=tuple(
            tuple(fmpq(base**count) for count in range(domain_size + 1))
            for base in (2, 3, 5)
        ),
        r_matrix=tuple(tuple(one for _ in range(3)) for _ in range(3)),
        graph_weight=one,
    )
    plan = build_boundary_profile_plan(component, domain_size, arithmetic)

    actual, _stats = evaluate_boundary_profile(
        component,
        plan,
        domain_size,
        arithmetic,
    )

    assert plan.nodes[plan.root].kind is BPNodeKind.INDEPENDENT
    assert actual == fmpq((2 + 3 + 5) ** domain_size)


def test_independent_root_closing_handles_interactions_with_hard_tail():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    domain_size = 5
    component = BoundaryProfileComponent(
        cell_weights=tuple(fmpq(base) for base in (2, 3, 5)),
        w_tables=tuple(
            tuple(fmpq(base**count) for count in range(domain_size + 1))
            for base in (2, 3, 5)
        ),
        r_matrix=tuple(
            tuple(fmpq(value) for value in row)
            for row in (
                (1, 2, 3),
                (2, 1, 5),
                (3, 5, 1),
            )
        ),
        graph_weight=fmpq(1),
    )
    plan = build_boundary_profile_plan(component, domain_size, arithmetic)

    actual, stats = evaluate_boundary_profile(
        component,
        plan,
        domain_size,
        arithmetic,
    )

    assert plan.strategy.startswith("independent-")
    assert stats.independent_root_states > 0
    assert actual == _direct_master_sum(component, domain_size, arithmetic)


def test_symmetric_balanced_composition_matches_direct_sum():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    domain_size = 7
    one = arithmetic.one()
    three = arithmetic.from_int(3)
    component = BoundaryProfileComponent(
        cell_weights=tuple(fmpq(2) for _ in range(5)),
        w_tables=tuple(
            tuple(fmpq(2**count) for count in range(domain_size + 1))
            for _ in range(5)
        ),
        r_matrix=tuple(
            tuple(one if left == right else three for right in range(5))
            for left in range(5)
        ),
        graph_weight=one,
    )
    plan = build_boundary_profile_plan(component, domain_size, arithmetic)

    actual, _stats = evaluate_boundary_profile(
        component,
        plan,
        domain_size,
        arithmetic,
    )

    assert any(node.kind is BPNodeKind.SYMMETRIC for node in plan.nodes)
    assert actual == _direct_master_sum(component, domain_size, arithmetic)


def test_disconnected_join_keeps_general_parent_projection():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    domain_size = 5
    bases = (2, 3, 5, 7)
    diagonals = (11, 13, 17, 19)
    component = BoundaryProfileComponent(
        cell_weights=tuple(fmpq(base) for base in bases),
        w_tables=tuple(
            tuple(
                fmpq(base**count * diagonal ** (count * (count - 1) // 2))
                for count in range(domain_size + 1)
            )
            for base, diagonal in zip(bases, diagonals)
        ),
        r_matrix=tuple(
            tuple(fmpq(value) for value in row)
            for row in (
                (11, 2, 1, 1),
                (2, 13, 1, 1),
                (1, 1, 17, 3),
                (1, 1, 3, 19),
            )
        ),
        graph_weight=fmpq(1),
    )
    plan = build_boundary_profile_plan(component, domain_size, arithmetic)
    root = plan.nodes[plan.root]

    actual, _stats = evaluate_boundary_profile(
        component,
        plan,
        domain_size,
        arithmetic,
    )

    assert root.kind is BPNodeKind.JOIN
    assert not root.cross_nonunit
    assert root.classes == ((0, 1, 2, 3),)
    assert root.left_projection == (0,)
    assert root.right_projection == (0,)
    assert actual == _direct_master_sum(component, domain_size, arithmetic)


def test_polynomial_kernel_matches_direct_sum_and_respects_degree_limit():
    arithmetic = ArithmeticContext(
        ArithmeticBackend.FMPQ_POLY,
        ("x",),
        degree_limits=(("x", 3),),
    )
    domain_size = 3
    one = arithmetic.one()
    x = arithmetic.symbol("x")
    component = BoundaryProfileComponent(
        cell_weights=(x, one),
        w_tables=(
            tuple(arithmetic.power(x, count) for count in range(domain_size + 1)),
            tuple(one for _ in range(domain_size + 1)),
        ),
        r_matrix=((one, x), (x, one)),
        graph_weight=one,
    )
    plan = build_boundary_profile_plan(component, domain_size, arithmetic)

    actual, _stats = evaluate_boundary_profile(
        component,
        plan,
        domain_size,
        arithmetic,
    )

    assert actual == _direct_master_sum(component, domain_size, arithmetic)
    assert actual.degree() <= 3


@pytest.mark.parametrize("domain_size", (0, 3))
def test_empty_cell_set_has_only_the_empty_domain_model(domain_size):
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    component = BoundaryProfileComponent(
        cell_weights=(),
        w_tables=(),
        r_matrix=(),
        graph_weight=fmpq(1),
    )
    plan = build_boundary_profile_plan(component, domain_size, arithmetic)

    actual, _stats = evaluate_boundary_profile(
        component,
        plan,
        domain_size,
        arithmetic,
    )

    assert actual == (1 if domain_size == 0 else 0)
