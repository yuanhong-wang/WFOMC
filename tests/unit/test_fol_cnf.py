from __future__ import annotations

from itertools import product

from wfomc.fol import FOLContext, evaluate
from wfomc.fol.cnf import encode_tseitin


def test_encode_tseitin_preserves_original_atom_mapping_and_auxiliary_range():
    ctx = FOLContext()
    x = ctx.variable("X")
    p = ctx.predicate("P", 1)
    q = ctx.predicate("Q", 1)
    formula = p(x) & (q(x) | ~p(x))

    cnf = encode_tseitin(formula)

    assert cnf.atoms == (p(x), q(x))
    assert cnf.atom_to_var == {p(x): 1, q(x): 2}
    assert cnf.n_vars == 5
    assert cnf.auxiliary_vars == frozenset({3, 4, 5})
    assert cnf.clauses[-1] == (5,)


def test_tseitin_auxiliaries_have_one_extension_per_satisfying_atom_assignment():
    ctx = FOLContext()
    x = ctx.variable("X")
    p = ctx.predicate("P", 1)
    q = ctx.predicate("Q", 1)
    formula = ctx.iff(p(x), q(x) | ~p(x))
    cnf = encode_tseitin(formula)

    for atom_values in product((False, True), repeat=len(cnf.atoms)):
        atom_assignment = dict(zip(cnf.atoms, atom_values))
        extensions = 0
        for aux_values in product((False, True), repeat=len(cnf.auxiliary_vars)):
            values = dict(
                zip(
                    (*range(1, len(cnf.atoms) + 1), *sorted(cnf.auxiliary_vars)),
                    (*atom_values, *aux_values),
                )
            )
            if all(
                any(values[abs(literal)] == (literal > 0) for literal in clause)
                for clause in cnf.clauses
            ):
                extensions += 1
        assert extensions == int(evaluate(formula, atom_assignment))


def test_tseitin_reuses_repeated_non_atomic_subformulas():
    ctx = FOLContext()
    x = ctx.variable("X")
    p = ctx.predicate("P", 1)
    q = ctx.predicate("Q", 1)
    shared = p(x) | q(x)

    cnf = encode_tseitin(ctx.conjunction(shared, shared))

    assert cnf.n_vars == 4
    assert cnf.auxiliary_vars == frozenset({3, 4})
