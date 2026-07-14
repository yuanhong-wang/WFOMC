"""Focused contracts for the solver-specific C2 normal form."""

from __future__ import annotations

import pytest

from wfomc.fol import FOLContext, Iff, Not, predicates
from wfomc.fol.normal_form import C2NormalForm, normalize, validate_normal_form
from wfomc.fol.normal_form.c2.norm_form import (
    CountDefinition,
    CountSection,
    ForallCountSection,
)
from wfomc.fol.normal_form.c2.normalize import NormalizeError
from wfomc.fol.normal_form.c2.validation import NormalFormValidationError
from wfomc.parser import parse_formula


def test_normalize_splits_top_level_c2_sections():
    ctx = FOLContext()
    x, y = ctx.vars("X Y")
    p = ctx.predicate("P", 1)
    q = ctx.predicate("Q", 1)
    u = ctx.predicate("U", 1)
    edge = ctx.predicate("E", 2)
    source = ctx.conjunction(
        ctx.forall(x, p(x)),
        ctx.exists(x, q(x)),
        ctx.count(x, "=", 1, u(x)),
        ctx.forall(x, ctx.count(y, "=", 2, edge(x, y))),
    )

    normal_form = normalize(source)

    assert normal_form.qf_formula == p(x)
    assert normal_form.exists == (ctx.exists(x, q(x)),)
    assert normal_form.forall_exists == ()
    assert isinstance(normal_form.counts[0], CountSection)
    assert normal_form.counts[0].body == u(x)
    assert isinstance(normal_form.forall_counts[0], ForallCountSection)
    assert normal_form.forall_counts[0].body == edge(x, y)
    assert normal_form.count_definitions == ()


def test_normalize_simplifies_zero_and_positive_counts():
    global_zero = normalize(parse_formula(r"\exists_=0 X: U(X)"))
    row_zero = normalize(parse_formula(r"\forall X: (\exists_=0 Y: R(X,Y))"))
    global_positive = normalize(parse_formula(r"\exists_>0 X: U(X)"))
    row_positive = normalize(
        parse_formula(r"\forall X: (\exists_>=1 Y: R(X,Y))")
    )

    assert str(global_zero.qf_formula) == r"~U(X)"
    assert global_zero.counts == ()
    assert str(row_zero.qf_formula) == r"~R(X,Y)"
    assert row_zero.forall_counts == ()
    assert str(global_positive.exists[0]) == r"\exists X: U(X)"
    assert str(row_positive.forall_exists[0]) == r"\forall X: \exists Y: R(X,Y)"


def test_non_atomic_count_body_gets_relation_abstraction():
    normal_form = normalize(parse_formula(r"\exists_=2 X: (P(X) & Q(X))"))

    assert str(normal_form.counts[0].body).startswith("@c2_rel_")
    assert "<->" in str(normal_form.qf_formula)
    assert normal_form.count_definitions == ()


def test_nested_universal_bodies_get_scott_abstraction():
    existential = normalize(parse_formula(r"\exists X: (\forall Y: E(X,Y))"))
    counted = normalize(parse_formula(r"\exists_=2 X: (\forall Y: E(X,Y))"))

    assert str(existential.exists[0]).startswith(r"\exists X: @c2_quant_")
    assert len(existential.forall_exists) == 1
    assert str(counted.counts[0].body).startswith("@c2_quant_")
    assert len(counted.forall_exists) == 1


def test_embedded_counts_become_explicit_definitions():
    closed = normalize(
        parse_formula(r"(\exists_<=2 X: D(X)) -> (\forall X: U(X))")
    )
    row = normalize(
        parse_formula(r"\forall X: (P(X) | (\exists_{1mod3} Y: R(X,Y)))")
    )
    negated = normalize(parse_formula(r"~(\exists_{1mod3} X: P(X))"))

    assert isinstance(closed.count_definitions[0].section, CountSection)
    assert closed.count_definitions[0].marker.predicate.arity == 0
    assert isinstance(row.count_definitions[0].section, ForallCountSection)
    assert row.count_definitions[0].section.count == (1, 3)
    assert isinstance(negated.qf_formula, Not)
    assert negated.qf_formula.body == negated.count_definitions[0].marker


def test_iff_reuses_one_count_definition():
    ctx = FOLContext()
    x = ctx.variable("X")
    counted = ctx.count(x, "=", 2, ctx.predicate("P", 1)(x))

    normal_form = normalize(ctx.iff(counted, ctx.predicate("Q", 0)()))

    assert isinstance(normal_form.qf_formula, Iff)
    assert len(normal_form.count_definitions) == 1
    assert normal_form.qf_formula.left == normal_form.count_definitions[0].marker


def test_mod_counts_keep_typed_global_and_row_sections():
    global_count = normalize(parse_formula(r"\exists_{1mod3} X: P(X)"))
    row_count = normalize(
        parse_formula(r"\forall X: (\exists_{1mod3} Y: R(X,Y))")
    )

    validate_normal_form(global_count)
    validate_normal_form(row_count)
    assert (global_count.counts[0].comparator, global_count.counts[0].count) == (
        "mod",
        (1, 3),
    )
    assert row_count.forall_counts[0].comparator == "mod"


def test_generated_names_avoid_predicate_and_variable_collisions():
    ctx = FOLContext()
    x = ctx.variable("X")
    y = ctx.variable("Y")
    existing_variable = ctx.variable("X_c2_0")
    user_marker = ctx.predicate("@c2_quant_0", 1)
    edge = ctx.predicate("E", 2)
    with_predicate_collision = ctx.forall(
        x, ctx.disjunction(user_marker(x), ctx.exists(y, edge(x, y)))
    )
    with_variable_collision = ctx.forall(
        x,
        ctx.forall(
            existing_variable,
            ctx.exists(x, edge(x, existing_variable)),
        ),
    )

    first = normalize(with_predicate_collision)
    second = normalize(with_variable_collision)

    first_names = {predicate.name for predicate in predicates(first.qf_formula)}
    second_markers = {
        predicate.name
        for predicate in predicates(second.qf_formula)
        if predicate.name.startswith("@c2_quant_")
    }
    assert {"@c2_quant_0", "@c2_quant_1"} <= first_names
    assert second_markers == {"@c2_quant_0"}
    assert second.requires_nonempty_domain


def test_normalize_rejects_formulas_outside_the_solver_fragment():
    ctx = FOLContext()
    x, y, z = ctx.vars("X Y Z")
    edge = ctx.predicate("R", 2)
    cases = (
        (
            ctx.forall((x, y), edge(x, y)),
            "Multi-variable quantifier",
        ),
        (
            ctx.forall(
                x,
                ctx.forall(y, ctx.forall(z, edge(x, y) & edge(y, z))),
            ),
            "more than two variables",
        ),
        (ctx.predicate("T", 3)(x, x, x), "arity at most 2"),
        (ctx.exists(x, edge(x, y)), "closed sentence.*Y"),
    )

    for source, message in cases:
        with pytest.raises(NormalizeError, match=message):
            normalize(source)


def test_validate_accepts_a_complete_typed_normal_form():
    ctx = FOLContext()
    x, y = ctx.vars("X Y")
    unary = ctx.predicate("U", 1)(x)
    row = ctx.predicate("R", 2)(x, y)
    section = CountSection("<=", 2, unary, x)
    normal_form = C2NormalForm(
        forall_counts=(ForallCountSection("mod", (1, 3), row, x, y),),
        counts=(section,),
        count_definitions=(CountDefinition(ctx.predicate("M", 0)(), section),),
    )

    validate_normal_form(normal_form)


def test_validate_rejects_malformed_normal_forms():
    ctx = FOLContext()
    x, y, z = ctx.vars("X Y Z")
    unary = ctx.predicate("U", 1)(x)
    row = ctx.predicate("R", 2)(x, y)
    marker = ctx.predicate("M", 0)()
    cases = (
        (
            C2NormalForm(
                forall_counts=(ForallCountSection("~", 2, row, x, y),)
            ),
            "unsupported comparator",
        ),
        (
            C2NormalForm(
                forall_counts=(ForallCountSection("mod", (3, 3), row, x, y),)
            ),
            "0 <= remainder < modulus",
        ),
        (
            C2NormalForm(
                forall_counts=(
                    ForallCountSection("=", 2, ctx.predicate("S", 2)(x, z), x, y),
                )
            ),
            "scoped variables",
        ),
        (
            C2NormalForm(
                count_definitions=(
                    CountDefinition(None, CountSection("=", 1, unary, x)),  # type: ignore[arg-type]
                )
            ),
            "marker must be a typed Atom",
        ),
        (
            C2NormalForm(qf_formula=ctx.exists(x, unary)),
            "quantifier-free",
        ),
        (
            C2NormalForm(
                counts=(CountSection("=", 1, ctx.predicate("P", 1)(ctx.constant("a")), ctx.constant("a")),)  # type: ignore[arg-type]
            ),
            "counted_var.*Variable",
        ),
        (
            C2NormalForm(exists=(ctx.exists(x, ctx.predicate("F", 2)(x, y)),)),
            "unbound variables.*Y",
        ),
        (
            C2NormalForm(
                count_definitions=(
                    CountDefinition(marker, CountSection("=", 1, unary, x)),
                    CountDefinition(marker, CountSection("=", 2, unary, x)),
                )
            ),
            "duplicate marker",
        ),
        (
            C2NormalForm(
                count_definitions=(
                    CountDefinition(
                        ctx.predicate("N", 1)(x),
                        ForallCountSection("=", 1, ctx.predicate("D", 2)(x, x), x, x),
                    ),
                )
            ),
            "outer_var and counted_var must be different",
        ),
    )

    for malformed, message in cases:
        with pytest.raises(NormalFormValidationError, match=message):
            validate_normal_form(malformed)


def test_normalize_validates_existing_normal_form():
    ctx = FOLContext()
    x = ctx.variable("X")
    malformed = C2NormalForm(
        counts=(CountSection("=", 1, "P(X)", x),)  # type: ignore[arg-type]
    )

    with pytest.raises(NormalFormValidationError, match="body must be a typed Atom"):
        normalize(malformed)
