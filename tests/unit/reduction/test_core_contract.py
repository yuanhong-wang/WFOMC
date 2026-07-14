"""Tests for the pure reduction branch contract."""

from __future__ import annotations

import pytest

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.problem import CompiledProblem, Problem, ReducedProblem
from wfomc.reduction import (
    begin_reduction,
    compose_decoders,
)


def _arithmetic() -> ArithmeticContext:
    return ArithmeticContext(ArithmeticBackend.FMPQ)


def _problem() -> ReducedProblem:
    from wfomc.fol import true

    return begin_reduction(
        Problem(sentence=true(), domain=frozenset({"a", "b"}))
    )


def test_begin_reduction_creates_stable_c2_stage():
    from wfomc.fol.normal_form import C2NormalForm

    reduced = _problem()

    assert isinstance(reduced, ReducedProblem)
    assert isinstance(reduced.normal_form, C2NormalForm)


def test_begin_reduction_rejects_empty_domain_for_scott_abstraction():
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

    with pytest.raises(ValueError, match="non-empty domain"):
        begin_reduction(Problem(sentence=source, domain=frozenset()))


def test_begin_reduction_reserves_weight_only_predicate_names():
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

    reduced = begin_reduction(
        Problem(
            sentence=source,
            domain=frozenset({"a"}),
            weights={reserved: (2, 1)},
        )
    )
    names = {
        predicate.name for predicate in predicates(reduced.normal_form.qf_formula)
    }

    assert "@c2_quant_0" not in names
    assert "@c2_quant_1" in names


def test_problem_stages_reject_wrong_formula_representation():
    from wfomc.fol import Variable, forall, true
    from wfomc.fol.normal_form import C2NormalForm

    with pytest.raises(TypeError, match="Problem.sentence"):
        Problem(sentence=C2NormalForm())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ReducedProblem.normal_form"):
        ReducedProblem(normal_form=true())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quantifier-free"):
        CompiledProblem(
            sentence=forall(Variable("X"), true()),
            arithmetic=_arithmetic(),
        )


def test_problem_rejects_invalid_circular_order_size():
    from wfomc.fol import true

    with pytest.raises(ValueError, match="circular_order_size"):
        Problem(
            sentence=true(),
            domain=frozenset(("a", "b")),
            circular_order_size=3,
        )

    assert Problem(
        sentence=true(),
        domain=frozenset(("a", "b")),
        circular_order_size=0,
    ).circular_order_size == 0


def test_compose_decoders_runs_inner_then_outer():
    decoder = compose_decoders(
        lambda value, **_: value * 2,
        lambda value, **_: value + 1,
    )
    assert decoder(3) == 8


def test_apply_reductions_decodes_in_reverse_transformation_order():
    from wfomc.algo import AlgoOptions
    from wfomc.fol import true
    from wfomc.reduction import ProblemWithDecoder, apply_reductions

    def multiply(problem, *, options):
        return ProblemWithDecoder(problem, lambda value, **_: value * 2)

    def increment(problem, *, options):
        return ProblemWithDecoder(problem, lambda value, **_: value + 1)

    branch = apply_reductions(
        Problem(
            sentence=true(),
            domain=frozenset({"a"}),
        ),
        (multiply, increment),
        AlgoOptions(),
    ).expect_single()

    # Transformations run multiply then increment, so decoding must undo them
    # in reverse: multiply(increment(raw)).
    assert branch.decoder(3) == 8


def test_counting_reduction_uses_fresh_internal_predicates():
    from wfomc.algo import AlgoOptions
    from wfomc.parser import parse_problem
    from wfomc.reduction import reduce_counting_quantifiers

    problem = parse_problem(
        r"""
\forall X: (\exists_{=2} Y: (R1(X,Y))) &
\forall X: (\exists_{=2} Y: (R2(X,Y)))
domain = 3
"""
    )
    reduced = reduce_counting_quantifiers(
        begin_reduction(problem), options=AlgoOptions()
    ).problem

    assert len(
        {
            predicate.name
            for predicate in reduced.weights
            if predicate.name.startswith("__sk_")
        }
    ) == 4


def test_existential_reduction_uses_fresh_collision_free_predicates():
    from wfomc.algo import AlgoOptions
    from wfomc.fol import FOLContext
    from wfomc.reduction import reduce_existential_quantifiers

    fol = FOLContext()
    x, y = fol.vars("X Y")
    first = fol.predicate("R1", 2)
    second = fol.predicate("R2", 2)
    existing = fol.predicate("__skolem0", 1)
    problem = begin_reduction(
        Problem(
            sentence=fol.conjunction(
                fol.forall(x, fol.exists(y, first(x, y))),
                fol.forall(x, fol.exists(y, second(x, y))),
            ),
            domain=frozenset((fol.constant("a"), fol.constant("b"))),
            weights={existing: (1, 1)},
        )
    )

    reduced = reduce_existential_quantifiers(problem, options=AlgoOptions())
    generated = {
        predicate.name
        for predicate in reduced.weights
        if predicate.name.startswith("__skolem")
    }

    assert generated == {"__skolem0", "__skolem1", "__skolem2"}
