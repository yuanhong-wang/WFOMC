"""Differential checks for cell-graph factorization semantics."""

from __future__ import annotations

import pytest

from wfomc import AlgoName, Domain, Problem, ProblemInstance, solve
from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.cell_graph import build_cell_graphs
from wfomc.fol import FOLContext, Formula


@pytest.mark.parametrize("domain_size", [1, 2])
def test_standard_matches_grounding_for_cross_element_unary_constraint(
    domain_size: int,
):
    ctx = FOLContext()
    x, y = ctx.vars("X Y")
    predicate = ctx.predicate("P", 1)
    sentence = ctx.forall(
        x,
        ctx.forall(y, ctx.iff(predicate(x), predicate(y))),
    )
    problem = _problem(ctx, sentence, domain_size)

    assert _count(problem, AlgoName.STANDARD) == _count(problem, AlgoName.PROPOSITIONAL)


@pytest.mark.parametrize("domain_size", [1, 2])
def test_standard_matches_grounding_when_nullary_branch_removes_predicate(
    domain_size: int,
):
    ctx = FOLContext()
    x = ctx.variable("X")
    switch = ctx.predicate("A", 0)
    predicate = ctx.predicate("P", 1)
    sentence = ctx.forall(x, ctx.disjunction(switch(), predicate(x)))
    problem = _problem(
        ctx,
        sentence,
        domain_size,
        weights={switch: (2, 3), predicate: (5, 7)},
    )

    assert _count(problem, AlgoName.STANDARD) == _count(problem, AlgoName.PROPOSITIONAL)


@pytest.mark.parametrize("domain_size", [1, 2])
def test_standard_preserves_binary_vocabulary_across_nullary_branches(
    domain_size: int,
):
    ctx = FOLContext()
    x, y = ctx.vars("X Y")
    switch = ctx.predicate("A", 0)
    relation = ctx.predicate("R", 2)
    sentence = ctx.forall(
        x,
        ctx.forall(y, ctx.disjunction(switch(), relation(x, y))),
    )
    problem = _problem(ctx, sentence, domain_size)

    assert _count(problem, AlgoName.STANDARD) == _count(problem, AlgoName.PROPOSITIONAL)


def test_nullary_branches_keep_original_non_nullary_predicate_universe():
    ctx = FOLContext()
    x = ctx.variable("X")
    switch = ctx.predicate("A", 0)
    predicate = ctx.predicate("P", 1)
    sentence = ctx.disjunction(switch(), predicate(x))
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)

    branches = tuple(build_cell_graphs(sentence, {}, arithmetic))

    assert len(branches) == 2
    assert sorted(len(data.cells) for data, _weight in branches) == [1, 2]
    assert all(
        all(predicate in cell.preds for cell in data.cells)
        for data, _weight in branches
    )
    assert {
        dict(data.nullary_assignments)[switch]
        for data, _weight in branches
    } == {False, True}


def test_standard_counts_the_empty_structure_for_true_sentence():
    ctx = FOLContext()
    problem = ProblemInstance(Problem(sentence=ctx.true()), Domain())

    assert _count(problem, AlgoName.STANDARD) == 1
    assert _count(problem, AlgoName.STANDARD) == _count(problem, AlgoName.PROPOSITIONAL)


def test_standard_rejects_closed_false_sentence_on_empty_domain():
    ctx = FOLContext()

    assert (
        _count(
            ProblemInstance(Problem(sentence=ctx.false()), Domain()),
            AlgoName.STANDARD,
        )
        == 0
    )


def test_standard_treats_unary_contradiction_as_vacuously_true_on_empty_domain():
    ctx = FOLContext()
    x = ctx.variable("X")
    predicate = ctx.predicate("P", 1)
    sentence = ctx.forall(
        x,
        ctx.conjunction(predicate(x), ctx.neg(predicate(x))),
    )

    assert (
        _count(
            ProblemInstance(Problem(sentence=sentence), Domain()),
            AlgoName.STANDARD,
        )
        == 1
    )


def test_overlapping_cell_profiles_do_not_multiply_pair_weights():
    ctx = FOLContext()
    x, y = ctx.vars("X Y")
    profile = ctx.predicate("Profile", 1)
    relation = ctx.predicate("R", 2)
    sentence = relation(x, y) | ~relation(x, y)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)

    baseline, _ = next(
        build_cell_graphs(
            sentence,
            {},
            arithmetic,
            required_unary_preds=frozenset((profile,)),
        )
    )
    overlapping, _ = next(
        build_cell_graphs(
            sentence,
            {},
            arithmetic,
            cell_formulas=(profile(x), ctx.true()),
        )
    )

    assert set(overlapping.cells) == set(baseline.cells)
    baseline_weights = {
        (left, right): baseline.pair_factors[left_idx][right_idx].total_weight
        for left_idx, left in enumerate(baseline.cells)
        for right_idx, right in enumerate(baseline.cells)
    }
    overlapping_weights = {
        (left, right): overlapping.pair_factors[left_idx][right_idx].total_weight
        for left_idx, left in enumerate(overlapping.cells)
        for right_idx, right in enumerate(overlapping.cells)
    }
    assert overlapping_weights == baseline_weights


def _problem(
    ctx: FOLContext,
    sentence: Formula,
    domain_size: int,
    *,
    weights: dict[object, tuple[object, object]] | None = None,
) -> ProblemInstance:
    return ProblemInstance(
        Problem(
            sentence=sentence,
            weights={} if weights is None else weights,
        ),
        Domain(
            frozenset(
                ctx.constant(f"d{index}") for index in range(domain_size)
            )
        ),
    )


def _count(problem: ProblemInstance, algorithm: AlgoName) -> int:
    return int(solve(problem, algo=algorithm))
