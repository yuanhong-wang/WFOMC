from __future__ import annotations

import pytest
from lark import UnexpectedInput

from wfomc.fol import (
    And,
    Atom,
    FOLContext,
    Formula,
    FormulaKind,
    Implies,
    Not,
    Quantifier,
    count,
    forall,
    free_vars,
    to_nnf,
)
from wfomc.parser.transformers.fol import parse


def test_context_hash_conses_symbols_and_formula_nodes():
    ctx = FOLContext()
    (x,) = ctx.vars("X")
    same_x = ctx.variable("X")
    predicate = ctx.predicate("P", 1)

    first = predicate(x)
    second = predicate(same_x)

    assert x is same_x
    assert first is second
    assert first.node_id == second.node_id
    assert isinstance(first, Atom)


def test_nary_connectives_are_flattened_and_keep_compat_args():
    ctx = FOLContext()
    x, y = ctx.vars("X Y")
    p = ctx.predicate("P", 1)
    q = ctx.predicate("Q", 1)
    r = ctx.predicate("R", 1)

    formula = ctx.conjunction(p(x), ctx.conjunction(q(x), r(y)))

    assert isinstance(formula, And)
    assert formula.op == FormulaKind.AND
    assert len(formula.args) == 3
    assert free_vars(formula) == frozenset({x, y})
    assert ("free_vars", formula.node_id) in ctx._analysis_cache


def test_parser_outputs_typed_node_classes():
    formula = parse(r"\forall X: (P(X) -> Q(X))")

    assert isinstance(formula, Quantifier)
    assert formula.op == FormulaKind.FORALL
    assert isinstance(formula.args[1], Implies)


def test_parser_accepts_atomic_quantifier_body_without_fallback():
    formula = parse(r"\forall X: P(X)")

    assert isinstance(formula, Quantifier)
    assert formula.op == FormulaKind.FORALL
    assert str(formula) == r"\forall X: P(X)"


def test_parser_accepts_canonical_counting_quantifier_forms():
    cases = (
        (r"\exists_{=2} X: (P(X))", "=", 2),
        (r"\exists_=2 X: (P(X))", "=", 2),
        (r"\exists_{<=2} X: (P(X))", "<=", 2),
        (r"\exists_<=2 X: (P(X))", "<=", 2),
        (r"\exists_{1mod3} X: (P(X))", "mod", (1, 3)),
        (r"\exists_1mod3 X: (P(X))", "mod", (1, 3)),
        (r"\exists_{1 mod 3} X: (P(X))", "mod", (1, 3)),
    )

    for source, comparator, count_param in cases:
        formula = parse(source)
        assert formula.op == FormulaKind.COUNT, source
        assert formula.args[1] == comparator, source
        assert formula.args[2] == count_param, source


def test_parser_requires_commas_between_terms():
    with pytest.raises(UnexpectedInput):
        parse(r"P(X Y)")


def test_parser_supports_nullary_predicates():
    bare = parse("P")
    called = parse("P()")

    assert isinstance(bare, Atom)
    assert isinstance(called, Atom)
    assert str(bare) == "P()"
    assert str(called) == "P()"


def test_parser_uses_one_context_for_repeated_symbols_and_nodes():
    formula = parse(r"\forall X: ((P(X)) & (P(X)))")
    body = formula.args[1]

    assert isinstance(body, And)
    assert body.args[0] is body.args[1]
    assert body.args[0].args[1] is body.args[1].args[1]
    assert body.args[0]._context is formula._context


def test_rewrite_eliminates_implications_and_pushes_negation():
    ctx = FOLContext()
    (x,) = ctx.vars("X")
    p = ctx.predicate("P", 1)
    q = ctx.predicate("Q", 1)

    formula = ctx.neg(ctx.implies(p(x), q(x)))
    nnf = to_nnf(formula)

    assert isinstance(nnf, And)
    assert all(arg.op != FormulaKind.IMPLIES for arg in nnf.args)
    assert any(isinstance(arg, Not) for arg in nnf.args)


def test_public_helpers_use_formula_context_when_available():
    ctx = FOLContext()
    x, y = ctx.vars("X Y")
    edge = ctx.predicate("E", 2)

    formula = forall(x, count(y, "=", 2, edge(x, y)))

    assert isinstance(formula, Formula)
    assert formula._context is ctx
    assert str(formula) == r"\forall X: \exists_{=2} Y: E(X,Y)"
