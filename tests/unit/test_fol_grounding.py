from __future__ import annotations

from itertools import product

import pytest

from wfomc.fol import FOLContext, atoms, evaluate, is_quantifier_free
from wfomc.fol.grounding import (
    at_least_k_clauses,
    at_most_k_clauses,
    exactly_k_clauses,
    ground_on_tuple,
    ground_qf_formula,
    ground_source_formula,
)


def test_ground_qf_formula_returns_ground_cnf_data():
    ctx = FOLContext()
    x = ctx.variable("X")
    predicate = ctx.predicate("P", 1)
    domain = [ctx.constant("a"), ctx.constant("b")]

    atom_to_id, id_to_predicate, clauses, unsatisfiable = ground_qf_formula(
        predicate(x),
        domain,
    )

    assert set(atom_to_id) == {predicate(domain[0]), predicate(domain[1])}
    assert set(id_to_predicate.values()) == {predicate}
    assert clauses
    assert not unsatisfiable


def test_ground_qf_formula_rejects_quantified_formula():
    ctx = FOLContext()
    x = ctx.variable("X")
    predicate = ctx.predicate("P", 1)

    with pytest.raises(ValueError, match="quantifier-free"):
        ground_qf_formula(
            ctx.forall(x, predicate(x)),
            [ctx.constant("a")],
        )


def test_ground_on_tuple_orders_free_variables_by_name():
    ctx = FOLContext()
    x, y = ctx.vars("X Y")
    relation = ctx.predicate("R", 2)
    first = ctx.constant("a")
    second = ctx.constant("b")

    grounded = ground_on_tuple(relation(y, x), first, second)

    assert set(atoms(grounded)) == {relation(second, first)}


def test_ground_source_formula_expands_nested_quantifiers_without_fo2_limit():
    ctx = FOLContext()
    x, y, z = ctx.vars("X Y Z")
    relation = ctx.predicate("R", 3)
    domain = tuple(ctx.constant(name) for name in ("a", "b"))
    source = ctx.forall(
        (x, y, z),
        relation(x, y, z) | ~relation(x, y, z),
    )

    grounded = ground_source_formula(source, domain)

    assert is_quantifier_free(grounded)
    assert len(atoms(grounded)) == len(domain) ** 3


def test_ground_source_formula_resolves_ground_equalities():
    ctx = FOLContext()
    x = ctx.variable("X")
    domain = tuple(ctx.constant(name) for name in ("a", "b"))

    grounded = ground_source_formula(ctx.exists(x, ctx.eq(x, domain[1])), domain)

    assert evaluate(grounded, {}) is True


@pytest.mark.parametrize(
    ("comparator", "threshold", "expected_models"),
    (
        ("=", 1, 2),
        ("!=", 1, 2),
        ("<", 1, 1),
        ("<=", 1, 3),
        (">", 1, 1),
        (">=", 1, 3),
        ("mod", (1, 2), 2),
    ),
)
def test_ground_source_formula_expands_all_counting_comparators(
    comparator: str,
    threshold: object,
    expected_models: int,
):
    ctx = FOLContext()
    x = ctx.variable("X")
    predicate = ctx.predicate("P", 1)
    domain = tuple(ctx.constant(name) for name in ("a", "b"))
    source = ctx.count(x, comparator, threshold, predicate(x))

    grounded = ground_source_formula(source, domain)
    ground_atoms = tuple(atoms(grounded))
    models = sum(
        evaluate(grounded, dict(zip(ground_atoms, values)))
        for values in product((False, True), repeat=len(ground_atoms))
    )

    assert is_quantifier_free(grounded)
    assert models == expected_models


def test_ground_source_formula_observes_empty_domain_quantifier_semantics():
    ctx = FOLContext()
    x = ctx.variable("X")
    predicate = ctx.predicate("P", 1)

    assert evaluate(ground_source_formula(ctx.forall(x, predicate(x)), ()), {})
    assert not evaluate(ground_source_formula(ctx.exists(x, predicate(x)), ()), {})
    assert evaluate(
        ground_source_formula(ctx.count(x, "=", 0, predicate(x)), ()),
        {},
    )


@pytest.mark.parametrize(
    "builder",
    (at_most_k_clauses, at_least_k_clauses, exactly_k_clauses),
)
@pytest.mark.parametrize("bound", (-1, 0, 1, 2, 3, 4))
def test_direct_cardinality_clauses_match_integer_semantics(builder, bound: int):
    literals = (1, 2, 3)
    clauses = builder(literals, bound)

    for values in product((False, True), repeat=len(literals)):
        actual = all(
            any(values[abs(literal) - 1] == (literal > 0) for literal in clause)
            for clause in clauses
        )
        count = sum(values)
        if builder is at_most_k_clauses:
            expected = count <= bound
        elif builder is at_least_k_clauses:
            expected = count >= bound
        else:
            expected = count == bound
        assert actual is expected, (builder.__name__, bound, values, clauses)
