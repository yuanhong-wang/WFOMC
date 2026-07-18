"""Tests for the domain-free reduction contract."""

from __future__ import annotations

import pytest

from wfomc.algo import AlgoOptions
from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.problem import CompiledBranchInstance, Domain, Problem
from wfomc.reduction import ReducedProblem, reduce_problem


def _arithmetic() -> ArithmeticContext:
    return ArithmeticContext(ArithmeticBackend.FMPQ)


def test_reduce_problem_creates_stable_c2_stage():
    from wfomc.fol import true
    from wfomc.fol.normal_form import C2NormalForm

    branches = reduce_problem(Problem(sentence=true()), AlgoOptions())

    assert len(branches) == 1
    assert isinstance(branches[0], ReducedProblem)
    assert isinstance(branches[0].normal_form, C2NormalForm)


def test_reduced_problem_records_nonempty_domain_requirement():
    from wfomc.fol import FOLContext

    ctx = FOLContext()
    x = ctx.variable("X")
    y = ctx.variable("Y")
    source = ctx.forall(
        x,
        ctx.disjunction(
            ctx.predicate("P", 1)(x),
            ctx.exists(y, ctx.predicate("R", 2)(x, y)),
        ),
    )

    reduced = reduce_problem(Problem(sentence=source), AlgoOptions())[0]

    assert reduced.min_domain_size == 1
    assert not reduced.applies(Domain())
    assert reduced.applies(Domain(frozenset({"a"})))


def test_reduce_problem_reserves_weight_only_predicate_names():
    from wfomc.fol import FOLContext, predicates

    ctx = FOLContext()
    x = ctx.variable("X")
    y = ctx.variable("Y")
    reserved = ctx.predicate("@c2_quant_0", 1)
    source = ctx.forall(
        x,
        ctx.disjunction(
            ctx.predicate("P", 1)(x),
            ctx.exists(y, ctx.predicate("R", 2)(x, y)),
        ),
    )

    reduced = reduce_problem(
        Problem(sentence=source, weights={reserved: (2, 1)}),
        AlgoOptions(),
    )[0]
    names = {predicate.name for predicate in predicates(reduced.normal_form.qf_formula)}

    assert "@c2_quant_0" not in names
    assert "@c2_quant_1" in names


def test_problem_stages_reject_wrong_formula_representation():
    from wfomc.fol import Variable, forall, true
    from wfomc.fol.normal_form import C2NormalForm

    with pytest.raises(TypeError, match="Problem.sentence"):
        Problem(sentence=C2NormalForm())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quantifier-free"):
        CompiledBranchInstance(
            sentence=forall(Variable("X"), true()),
            arithmetic=_arithmetic(),
        )


def test_problem_rejects_invalid_circular_order_size():
    with pytest.raises(ValueError, match="circular_order_size"):
        Domain(frozenset(("a", "b")), circular_order_size=3)

    assert (
        Domain(
            frozenset(("a", "b")),
            circular_order_size=0,
        ).circular_order_size
        == 0
    )


def test_counting_reduction_uses_fresh_internal_predicates():
    from wfomc.parser import parse_problem

    parsed = parse_problem(
        r"""
\forall X: (\exists_{=2} Y: (R1(X,Y))) &
\forall X: (\exists_{=2} Y: (R2(X,Y)))
domain = 3
"""
    )
    reduced = reduce_problem(parsed.problem, AlgoOptions())[0]

    assert (
        len(
            {
                predicate.name
                for predicate in reduced.weights
                if predicate.name.startswith("__sk_")
            }
        )
        == 4
    )


def test_existential_reduction_uses_fresh_collision_free_predicates():
    from wfomc.fol import FOLContext

    fol = FOLContext()
    x, y = fol.vars("X Y")
    first = fol.predicate("R1", 2)
    second = fol.predicate("R2", 2)
    existing = fol.predicate("__skolem0", 1)
    problem = Problem(
        sentence=fol.conjunction(
            fol.forall(x, fol.exists(y, first(x, y))),
            fol.forall(x, fol.exists(y, second(x, y))),
        ),
        weights={existing: (1, 1)},
    )

    reduced = reduce_problem(problem, AlgoOptions())[0]
    generated = {
        predicate.name
        for predicate in reduced.weights
        if predicate.name.startswith("__skolem")
    }

    assert generated == {"__skolem0", "__skolem1", "__skolem2"}
