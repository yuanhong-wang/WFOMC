"""Public result and solver-boundary regressions."""

from fractions import Fraction
from pathlib import Path

from flint import fmpq_mpoly_ctx

from wfomc import AlgoName, Problem, WFOMCResult, parse_input, parse_problem, solve
from wfomc.fol import FOLContext


ROOT = Path(__file__).parents[2]


def test_solve_returns_public_result_wrapper():
    problem = parse_input(str(ROOT / "models" / "2-colored-graph.wfomcs"))

    result = solve(problem, algo=AlgoName.FASTV2)

    assert isinstance(result, WFOMCResult)
    assert result.is_constant()
    assert result.constant_value() is not None


def test_result_exposes_projected_polynomial_terms():
    fol = FOLContext()
    variable = fol.variable("X")
    predicate = fol.predicate("P", 1)
    symbol_context = fmpq_mpoly_ctx.get(("x",), "lex")
    problem = Problem(
        sentence=fol.forall(variable, predicate(variable)),
        domain=frozenset((fol.constant("a"), fol.constant("b"))),
        weights={predicate: (symbol_context.gen(0), 1)},
    )

    result = solve(problem, algo=AlgoName.FASTV2)

    assert result.is_polynomial()
    assert dict(result.terms(("x",))) == {(2,): Fraction(1, 1)}


def test_unsatisfiable_problem_returns_zero():
    problem = parse_problem(
        r"""
\forall X: ((S(X) -> C(X)) & ~(S(X) & C(X)))
domain = {e_1, e_10, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9}

|C| = 5

S(e_1), S(e_10), S(e_2), S(e_3), S(e_4), S(e_5), S(e_6), S(e_7), S(e_8), S(e_9)
"""
    )

    result = solve(problem, algo=AlgoName.FASTV2)

    assert isinstance(result, WFOMCResult)
    assert result == 0
