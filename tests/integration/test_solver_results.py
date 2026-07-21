"""Public result and solver-boundary regressions."""

from fractions import Fraction
from math import comb
from pathlib import Path

import pytest
from flint import fmpq_mpoly_ctx

from wfomc import (
    AlgoName,
    CardinalityConstraints,
    CardinalityTerm,
    Comparator,
    Domain,
    LinearCardinalityConstraint,
    Problem,
    ProblemInstance,
    WFOMCResult,
    parse_problem_file,
    parse_problem,
    solve,
)
from wfomc.engine import compile_problem, instantiate_problem
from wfomc.fol import FOLContext


ROOT = Path(__file__).parents[2]


def _problem(*, domain, **kwargs) -> ProblemInstance:
    return ProblemInstance(Problem(**kwargs), Domain(frozenset(domain)))


def test_solve_returns_public_result_wrapper():
    problem = parse_problem_file(str(ROOT / "models" / "2-colored-graph.wfomcs"))

    result = solve(problem, algo=AlgoName.FASTV2)

    assert isinstance(result, WFOMCResult)
    assert result.is_constant()
    assert result.constant_value() is not None


def test_result_exposes_projected_polynomial_terms():
    fol = FOLContext()
    variable = fol.variable("X")
    predicate = fol.predicate("P", 1)
    symbol_context = fmpq_mpoly_ctx.get(("x",), "lex")
    problem = _problem(
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
    assert solve(problem, algo=AlgoName.BOUNDARY_PROFILE) == 0


@pytest.mark.parametrize(
    "algo",
    (
        AlgoName.STANDARD,
        AlgoName.FAST,
        AlgoName.FASTV2,
        AlgoName.INCREMENTAL,
        AlgoName.INCREMENTAL3,
        AlgoName.RECURSIVE,
        AlgoName.BOUNDARY_PROFILE,
    ),
)
def test_upper_cardinality_constraint_uses_truncated_polynomial_backend(algo):
    fol = FOLContext()
    variable = fol.variable("X")
    predicate = fol.predicate("P", 1)
    problem = _problem(
        sentence=fol.forall(variable, predicate(variable) | ~predicate(variable)),
        domain=frozenset(fol.constant(f"d{index}") for index in range(10)),
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    (CardinalityTerm(predicate),),
                    Comparator.LE,
                    3,
                ),
            )
        ),
    )

    assert solve(problem, algo=algo) == 176


def test_multiple_constraints_on_one_predicate_use_one_marker():
    fol = FOLContext()
    variable = fol.variable("X")
    predicate = fol.predicate("P", 1)
    domain_size = 5
    problem = _problem(
        sentence=fol.forall(variable, predicate(variable) | ~predicate(variable)),
        domain=frozenset(
            fol.constant(f"d{index}") for index in range(domain_size)
        ),
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    (CardinalityTerm(predicate),),
                    Comparator.GE,
                    1,
                ),
                LinearCardinalityConstraint(
                    (CardinalityTerm(predicate),),
                    Comparator.LE,
                    2,
                ),
            )
        ),
    )

    assert solve(problem, algo=AlgoName.FASTV2) == (
        comb(domain_size, 1) + comb(domain_size, 2)
    )


def test_negative_coefficient_uses_predicate_count_marker():
    fol = FOLContext()
    variable = fol.variable("X")
    predicate = fol.predicate("P", 1)
    domain_size = 3
    problem = _problem(
        sentence=fol.forall(variable, predicate(variable) | ~predicate(variable)),
        domain=frozenset(
            fol.constant(f"d{index}") for index in range(domain_size)
        ),
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    (CardinalityTerm(predicate, -1),),
                    Comparator.LE,
                    -1,
                ),
            )
        ),
    )

    assert solve(problem, algo=AlgoName.FASTV2) == 2**domain_size - 1


def test_binary_upper_cardinality_constraint_truncates_pair_weights():
    fol = FOLContext()
    left = fol.variable("X")
    right = fol.variable("Y")
    relation = fol.predicate("R", 2)
    domain_size = 4
    problem = _problem(
        sentence=fol.forall(
            left,
            fol.forall(right, relation(left, right) | ~relation(left, right)),
        ),
        domain=frozenset(
            fol.constant(f"d{index}") for index in range(domain_size)
        ),
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    (CardinalityTerm(relation),),
                    Comparator.LE,
                    2,
                ),
            )
        ),
    )

    assert solve(problem, algo=AlgoName.FASTV2) == sum(
        comb(domain_size**2, count) for count in range(3)
    )
    assert solve(problem, algo=AlgoName.BOUNDARY_PROFILE) == sum(
        comb(domain_size**2, count) for count in range(3)
    )


def test_joint_upper_cardinality_constraint_uses_one_linear_form_marker():
    fol = FOLContext()
    variable = fol.variable("X")
    first = fol.predicate("P", 1)
    second = fol.predicate("Q", 1)
    domain_size = 6
    problem = _problem(
        sentence=fol.forall(
            variable,
            (first(variable) | ~first(variable))
            & (second(variable) | ~second(variable)),
        ),
        domain=frozenset(
            fol.constant(f"d{index}") for index in range(domain_size)
        ),
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    (CardinalityTerm(first), CardinalityTerm(second)),
                    Comparator.LE,
                    2,
                ),
            )
        ),
    )

    compiled = compile_problem(problem.problem, algo=AlgoName.FASTV2)
    artifacts = instantiate_problem(compiled, problem.domain)
    assert len(artifacts.branches) == 1
    branch = artifacts.branches[0].problem

    cardinality_symbols = tuple(
        symbol
        for symbol in branch.internal_weight_symbols
        if symbol.startswith("__wfomc_cardinality_")
    )
    assert cardinality_symbols == ("__wfomc_cardinality_0",)
    assert solve(problem, algo=AlgoName.FASTV2) == sum(
        comb(2 * domain_size, count) for count in range(3)
    )
    assert solve(problem, algo=AlgoName.BOUNDARY_PROFILE) == sum(
        comb(2 * domain_size, count) for count in range(3)
    )


def test_canceling_terms_keep_shared_marker_degree_bound():
    fol = FOLContext()
    variable = fol.variable("X")
    canceled = fol.predicate("P", 1)
    counted = fol.predicate("Q", 1)
    domain_size = 3
    problem = _problem(
        sentence=fol.forall(
            variable,
            (canceled(variable) | ~canceled(variable))
            & (counted(variable) | ~counted(variable)),
        ),
        domain=frozenset(
            fol.constant(f"d{index}") for index in range(domain_size)
        ),
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    (
                        CardinalityTerm(canceled, 1),
                        CardinalityTerm(canceled, -1),
                        CardinalityTerm(counted, 1),
                    ),
                    Comparator.LE,
                    2,
                ),
            )
        ),
    )

    compiled = compile_problem(problem.problem, algo=AlgoName.FASTV2)
    artifacts = instantiate_problem(compiled, problem.domain)
    assert len(artifacts.branches) == 1
    branch = artifacts.branches[0].problem

    cardinality_limits = {
        symbol: limit
        for symbol, limit in branch.internal_weight_degree_limits
        if symbol.startswith("__wfomc_cardinality_")
    }
    assert set(cardinality_limits) == {"__wfomc_cardinality_0"}
    assert cardinality_limits["__wfomc_cardinality_0"].evaluate(domain_size) == 2
    assert solve(problem, algo=AlgoName.FASTV2) == (
        2**domain_size
        * sum(comb(domain_size, count) for count in range(3))
    )


def test_joint_cardinality_marker_tracks_term_coefficients():
    fol = FOLContext()
    variable = fol.variable("X")
    first = fol.predicate("P", 1)
    second = fol.predicate("Q", 1)
    domain_size = 3
    problem = _problem(
        sentence=fol.forall(
            variable,
            (first(variable) | ~first(variable))
            & (second(variable) | ~second(variable)),
        ),
        domain=frozenset(
            fol.constant(f"d{index}") for index in range(domain_size)
        ),
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    (
                        CardinalityTerm(first, 2),
                        CardinalityTerm(second, 3),
                    ),
                    Comparator.LE,
                    4,
                ),
            )
        ),
    )

    expected = sum(
        comb(domain_size, first_count) * comb(domain_size, second_count)
        for first_count in range(domain_size + 1)
        for second_count in range(domain_size + 1)
        if 2 * first_count + 3 * second_count <= 4
    )
    assert solve(problem, algo=AlgoName.FASTV2) == expected


def test_internal_cardinality_cap_does_not_truncate_colliding_user_symbol():
    fol = FOLContext()
    variable = fol.variable("X")
    weighted = fol.predicate("P", 1)
    bounded = fol.predicate("Q", 1)
    symbol_name = "__wfomc_cardinality_0"
    symbol_context = fmpq_mpoly_ctx.get((symbol_name,), "lex")
    domain_size = 4
    problem = _problem(
        sentence=fol.forall(
            variable,
            (weighted(variable) | ~weighted(variable))
            & (bounded(variable) | ~bounded(variable)),
        ),
        domain=frozenset(
            fol.constant(f"d{index}") for index in range(domain_size)
        ),
        weights={weighted: (symbol_context.gen(0), 1)},
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    (CardinalityTerm(bounded),),
                    Comparator.LE,
                    1,
                ),
            )
        ),
    )

    result = solve(problem, algo=AlgoName.FASTV2)

    assert dict(result.terms((symbol_name,))) == {
        (degree,): Fraction((domain_size + 1) * comb(domain_size, degree), 1)
        for degree in range(domain_size + 1)
    }
