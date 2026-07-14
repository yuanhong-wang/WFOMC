from __future__ import annotations

from wfomc.cell_graph.enumerate_cells import enumerate_cells, is_satisfiable
from wfomc.fol import FOLContext, model_literals
from wfomc.fol.cnf import encode_tseitin


def test_projected_sat_cells_match_native_models_with_auxiliaries():
    ctx = FOLContext()
    element = ctx.constant("c")
    p = ctx.predicate("P", 1)
    q = ctx.predicate("Q", 1)
    predicates = (p, q)
    formula = ctx.iff(p(element), q(element))

    actual = enumerate_cells(encode_tseitin(formula), predicates)
    expected = {
        tuple(
            {literal.pred: literal.positive for literal in model}[predicate]
            for predicate in predicates
        )
        for model in model_literals(formula)
    }

    assert {cell.code for cell in actual} == expected
    assert len(actual) == 2


def test_projected_sat_cells_include_free_cell_atoms():
    ctx = FOLContext()
    element = ctx.constant("c")
    constrained = ctx.predicate("P", 1)
    free = ctx.predicate("Free", 1)
    formula = constrained(element)

    cells = enumerate_cells(encode_tseitin(formula), (constrained, free))

    assert {cell.code for cell in cells} == {(True, False), (True, True)}


def test_projected_sat_cells_keep_missing_bits_in_predicate_order():
    ctx = FOLContext()
    element = ctx.constant("c")
    first_free = ctx.predicate("AFree", 1)
    constrained = ctx.predicate("P", 1)
    last_free = ctx.predicate("ZFree", 1)

    cells = enumerate_cells(
        encode_tseitin(constrained(element)),
        (first_free, constrained, last_free),
    )

    assert {cell.code for cell in cells} == {
        (False, True, False),
        (False, True, True),
        (True, True, False),
        (True, True, True),
    }


def test_projected_sat_cells_return_empty_for_unsatisfiable_formula():
    ctx = FOLContext()
    element = ctx.constant("c")
    predicate = ctx.predicate("P", 1)
    formula = predicate(element) & ~predicate(element)
    cnf = encode_tseitin(formula)

    assert not is_satisfiable(cnf)
    assert enumerate_cells(cnf, (predicate,)) == ()
