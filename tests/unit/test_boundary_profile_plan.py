from __future__ import annotations

from flint import fmpq

from wfomc.algo.boundary_profile.input import BoundaryProfileComponent
from wfomc.algo.boundary_profile.plan import (
    BPNodeKind,
    build_boundary_profile_plan,
)
from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext


def _component(
    local_bases: tuple[int, ...],
    interactions: tuple[tuple[int, ...], ...],
    domain_size: int,
) -> BoundaryProfileComponent:
    return BoundaryProfileComponent(
        cell_weights=tuple(fmpq(base) for base in local_bases),
        w_tables=tuple(
            tuple(fmpq(base**count) for count in range(domain_size + 1))
            for base in local_bases
        ),
        r_matrix=tuple(
            tuple(fmpq(value) for value in row) for row in interactions
        ),
        graph_weight=fmpq(1),
    )


def test_plan_boundary_classes_merge_without_splitting():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    component = _component(
        (2, 3, 5),
        (
            (1, 7, 11),
            (7, 1, 11),
            (11, 11, 1),
        ),
        3,
    )

    plan = build_boundary_profile_plan(component, 3, arithmetic)
    root = plan.nodes[plan.root]

    assert root.classes == ((0, 1, 2),)
    assert plan.bp_width >= 1
    for node in plan.nodes:
        if node.kind is not BPNodeKind.JOIN:
            continue
        assert node.left is not None and node.right is not None
        assert len(node.left_projection) == len(plan.nodes[node.left].classes)
        assert len(node.right_projection) == len(plan.nodes[node.right].classes)


def test_plan_selects_analytic_all_independent_block():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    component = _component(
        (2, 3, 5, 7),
        tuple(tuple(1 for _ in range(4)) for _ in range(4)),
        8,
    )

    plan = build_boundary_profile_plan(component, 8, arithmetic)

    assert plan.strategy == "all-independent"
    assert plan.nodes[plan.root].kind is BPNodeKind.INDEPENDENT
    assert plan.bp_width == 1
    assert {item.strategy for item in plan.candidate_estimates} >= {
        "all-independent",
        "tail-caterpillar",
        "greedy-agglomerative",
    }


def test_plan_exposes_identical_cells_as_symmetric_block():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    domain_size = 6
    component = _component(
        (2, 2, 2, 2, 2),
        tuple(
            tuple(1 if left == right else 3 for right in range(5))
            for left in range(5)
        ),
        domain_size,
    )

    plan = build_boundary_profile_plan(component, domain_size, arithmetic)

    assert any(node.kind is BPNodeKind.SYMMETRIC for node in plan.nodes)
    assert any(
        estimate.strategy == "components-symmetric"
        for estimate in plan.candidate_estimates
    )


def test_plan_accepts_unhashable_polynomial_interactions():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ_POLY, ("x",))
    one = arithmetic.one()
    x = arithmetic.symbol("x")
    component = BoundaryProfileComponent(
        cell_weights=(one, one),
        w_tables=((one, one, one), (one, one, one)),
        r_matrix=((one, x), (x, one)),
        graph_weight=one,
    )

    plan = build_boundary_profile_plan(component, 2, arithmetic)

    assert plan.nodes[plan.root].classes == ((0, 1),)
    assert plan.candidate_estimates


def test_structural_planning_does_not_infer_symmetry_from_small_domain_prefix():
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    component = BoundaryProfileComponent(
        cell_weights=(fmpq(2), fmpq(2)),
        w_tables=((fmpq(1), fmpq(2)), (fmpq(1), fmpq(2))),
        r_matrix=((fmpq(3), fmpq(1)), (fmpq(1), fmpq(5))),
        graph_weight=fmpq(1),
    )

    plan = build_boundary_profile_plan(component, 1, arithmetic)

    assert all(node.kind is not BPNodeKind.SYMMETRIC for node in plan.nodes)
