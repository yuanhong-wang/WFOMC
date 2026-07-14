from __future__ import annotations

from fractions import Fraction

import pytest
from flint import arb, arb_poly, fmpq_mpoly_ctx
from wfomc.algo import AlgoName, AlgoOptions, EvidenceStrategy, algo_spec
from wfomc.api import compile_problem, solve
from wfomc.arithmetic import ArithmeticBackend
from wfomc.engine.features import analyze_features
from wfomc.errors import ArithmeticBackendError
from wfomc.evidence import Evidence, GroundUnaryLiteral, UnaryEvidence
from wfomc.parser import parse_input, parse_problem
from wfomc.problem import Problem
from wfomc.weights import (
    WeightOptions,
    collect_symbolic_weight_variables,
)


def test_resolve_options_preserves_rounded_arithmetic():
    problem = parse_input("models/2-colored-graph.wfomcs")
    features = analyze_features(problem)
    requested = WeightOptions(precision="round", rounded_backend="float")

    resolved = algo_spec(AlgoName.STANDARD).resolve_options(
        features,
        AlgoOptions(weight_options=requested),
    )

    assert resolved.weight_options is requested


@pytest.mark.parametrize(
    ("backend", "expected_type"),
    (("float", float), ("arb", arb)),
)
def test_scalar_rounded_arithmetic_runs_end_to_end(backend, expected_type):
    problem = parse_input("models/2-colored-graph.wfomcs")

    result = solve(
        problem,
        algo=AlgoName.STANDARD,
        options=AlgoOptions(
            weight_options=WeightOptions(
                precision="round",
                rounded_backend=backend,
            )
        ),
    )

    assert isinstance(result.raw, expected_type)
    assert float(result) == pytest.approx(330626.0)


def test_incremental3_rounded_arithmetic_runs_end_to_end():
    problem = parse_problem(
        r"""
\forall X: (~E(X,X)) &
\forall X: (\forall Y: (E(X,Y) -> E(Y,X))) &
\forall X: (\exists_{=2} Y: (E(X,Y)))
domain = 7
"""
    )

    result = solve(
        problem,
        algo=AlgoName.INCREMENTAL3,
        options=AlgoOptions(
            weight_options=WeightOptions(
                precision="round",
                rounded_backend="arb",
            )
        ),
    )

    assert isinstance(result.raw, arb)
    assert float(result) == pytest.approx(465.0)


def test_rounded_arithmetic_flows_through_lifted_evidence():
    problem = parse_input("models/unary_evidence/evidence-only.wfomcs")

    result = solve(
        problem,
        algo=AlgoName.INCREMENTAL3,
        options=AlgoOptions(
            weight_options=WeightOptions(
                precision="round",
                rounded_backend="arb",
            )
        ),
    )

    assert isinstance(result.raw, arb)
    assert float(result) == pytest.approx(4.0)


def test_single_symbol_arb_polynomial_runs_end_to_end():
    from wfomc.fol import FOLContext, forall

    fol = FOLContext()
    variable = fol.variable("X")
    predicate = fol.predicate("P", 1)
    symbol_context = fmpq_mpoly_ctx.get(("w",), "lex")
    problem = Problem(
        sentence=forall(variable, predicate(variable) | ~predicate(variable)),
        domain=frozenset((fol.constant("a"), fol.constant("b"))),
        weights={predicate: (symbol_context.gen(0), 1)},
    )

    result = solve(
        problem,
        algo=AlgoName.FASTV2,
        options=AlgoOptions(
            weight_options=WeightOptions(
                precision="round",
                rounded_backend="arb",
            )
        ),
    )

    assert isinstance(result.raw, arb_poly)
    assert result.variable_names() == ("w",)
    assert str(result.raw).startswith("1.00000000000000*x^2")
    terms = tuple(result.terms())
    assert tuple(degrees for degrees, _ in terms) == ((0,), (1,), (2,))
    assert tuple(coefficient for _, coefficient in terms) == pytest.approx(
        (1.0, 2.0, 1.0)
    )


def test_rounded_cardinality_is_rejected_before_materialization():
    problem = parse_input("models/cardinality_constraints_example.wfomcs")

    with pytest.raises(
        ArithmeticBackendError,
        match="cardinality marker variables",
    ):
        solve(
            problem,
            algo=AlgoName.STANDARD,
            options=AlgoOptions(
                weight_options=WeightOptions(
                    precision="round",
                    rounded_backend="arb",
                )
            ),
        )


def test_propositional_reports_exact_only_external_backend():
    problem = parse_input("models/2-colored-graph.wfomcs")

    with pytest.raises(ArithmeticBackendError, match="Ganak"):
        solve(
            problem,
            algo=AlgoName.PROPOSITIONAL,
            options=AlgoOptions(
                weight_options=WeightOptions(
                    precision="round",
                    rounded_backend="arb",
                )
            ),
        )


def test_cardinality_reduction_injects_internal_weight_symbols():
    problem = parse_input("models/cardinality_constraints_example.wfomcs")
    assert collect_symbolic_weight_variables(problem) == ()

    artifacts = compile_problem(problem, algo=AlgoName.STANDARD)
    branch = artifacts.reduced_problem.expect_single().problem
    arithmetic = artifacts.algo_input.arithmetic

    assert branch.cardinality_constraints.is_empty
    assert branch.internal_weight_symbols
    assert set(branch.internal_weight_symbols) <= set(arithmetic.symbolic_variables)
    assert branch.internal_weight_degree_limits
    assert arithmetic.degree_limits == branch.internal_weight_degree_limits
    assert arithmetic.output_symbols == ()
    assert arithmetic.backend is ArithmeticBackend.FMPQ_POLY


def test_symbolic_weights_survive_evidence_cardinality_decoder_chain():
    from wfomc.fol import FOLContext, forall

    fol = FOLContext()
    variable = fol.variable("X")
    predicate = fol.predicate("P", 1)
    first = fol.constant("a")
    second = fol.constant("b")
    symbol_context = fmpq_mpoly_ctx.get(("w",), "lex")
    problem = Problem(
        sentence=forall(variable, predicate(variable) | ~predicate(variable)),
        domain=frozenset((first, second)),
        weights={predicate: (symbol_context.gen(0), 1)},
        evidence=Evidence(
            unary=UnaryEvidence((GroundUnaryLiteral(predicate, first, True),))
        ),
    )

    result = solve(
        problem,
        algo=AlgoName.FASTV2,
        options=AlgoOptions(evidence_strategy=EvidenceStrategy.CCS),
    )

    assert result.variable_names() == ("w",)
    assert tuple(sorted(result.terms())) == (
        ((1,), Fraction(1, 1)),
        ((2,), Fraction(1, 1)),
    )
