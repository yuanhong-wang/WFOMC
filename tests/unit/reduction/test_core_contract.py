"""Tests for the domain-free reduction contract."""

from __future__ import annotations

import pytest
from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.problem import Domain, Problem
from wfomc.reduction import reduce_problem
from wfomc.stages import CompiledBranchInstance, ReducedProblem


def _arithmetic() -> ArithmeticContext:
    return ArithmeticContext(ArithmeticBackend.FMPQ)


def test_reduce_problem_creates_stable_c2_stage():
    from wfomc.fol import true
    from wfomc.fol.normal_form import C2NormalForm

    reduced = reduce_problem(Problem(sentence=true()))

    assert isinstance(reduced, ReducedProblem)
    assert isinstance(reduced.normal_form, C2NormalForm)


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

    reduced = reduce_problem(Problem(sentence=source))

    assert reduced.min_domain_size == 1


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
    )
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
    reduced = reduce_problem(parsed.problem)

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


def test_counting_reduction_keeps_domain_terms_symbolic():
    import math

    from wfomc.stages import CardinalityDecoderSpec, DivideDecoderSpec
    from wfomc.parser import parse_problem

    parsed = parse_problem(
        r"""
\forall X: (\exists_{=2} Y: (R(X,Y)))
domain = 3
"""
    )

    reduced = reduce_problem(parsed.problem)
    divide = next(
        step
        for step in reduced.decoder_spec.steps
        if isinstance(step, DivideDecoderSpec)
    )
    cardinality = next(
        step
        for step in reduced.decoder_spec.steps
        if isinstance(step, CardinalityDecoderSpec)
    )
    row_constraint = cardinality.constraints[0]

    assert divide.coefficient.evaluate(2) == math.factorial(2) ** 2
    assert divide.coefficient.evaluate(5) == math.factorial(2) ** 5
    assert row_constraint.rhs.evaluate(2) == 4
    assert row_constraint.rhs.evaluate(5) == 10


def test_cardinality_reduction_preserves_raw_weights_until_compilation():
    from fractions import Fraction

    from wfomc.cardinality_constraints import (
        CardinalityConstraints,
        CardinalityTerm,
        Comparator,
        LinearCardinalityConstraint,
    )
    from wfomc.fol import FOLContext

    fol = FOLContext()
    predicate = fol.predicate("P", 1)
    raw_weights = (Fraction(2, 3), Fraction(5, 7))
    problem = Problem(
        sentence=fol.true(),
        weights={predicate: raw_weights},
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    terms=(CardinalityTerm(predicate),),
                    comparator=Comparator.EQ,
                    rhs=1,
                ),
            )
        ),
    )

    reduced = reduce_problem(problem)

    assert reduced.weights[predicate] == raw_weights
    assert reduced.internal_weight_symbols


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

    reduced = reduce_problem(problem)
    generated = {
        predicate.name
        for predicate in reduced.weights
        if predicate.name.startswith("__skolem")
    }

    assert generated == {"__skolem0", "__skolem1", "__skolem2"}
