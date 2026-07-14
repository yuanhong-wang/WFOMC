from __future__ import annotations

import pytest

from wfomc.fol import FOLContext, atoms
from wfomc.fol.grounding import ground_on_tuple, ground_qf_formula


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
