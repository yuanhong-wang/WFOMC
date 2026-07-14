from __future__ import annotations

import pytest

from wfomc.algo import (
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    LinearOrderEncoding,
    algo_spec,
)
from wfomc.algo.incremental.input import OrderedCellGraphInput
from wfomc.algo.core import compile_reduced_problem
from wfomc.algo.standard.input import build_input
from wfomc.algo.propositional.input import GroundCNFInput
from wfomc.algo.standard.input import StandardInput
from wfomc.engine import analyze_problem, compile_problem, solve
from wfomc.errors import UnsupportedFeatureError
from wfomc.parser import parse_input
from wfomc.problem import ReducedProblem
from wfomc.reduction import begin_reduction


def test_analyze_problem_stops_before_reduction_and_materialization():
    problem = parse_input("models/unary_evidence/evidence-only.wfomcs")

    artifacts = analyze_problem(problem)

    assert artifacts.parsed_problem is problem
    assert artifacts.feature_set is not None
    assert artifacts.reduced_problem is None
    assert artifacts.algo_inputs == ()
    assert artifacts.algo_input is None


def test_algorithm_specs_declare_maturity_and_external_requirements():
    assert algo_spec(AlgoName.STANDARD).maturity is AlgoMaturity.STABLE
    assert algo_spec(AlgoName.PROPOSITIONAL).maturity is AlgoMaturity.BETA
    assert algo_spec(AlgoName.PROPOSITIONAL).external_requirements == (
        "Ganak executable",
    )
    assert algo_spec(AlgoName.TAIL_SIGNATURE).maturity is AlgoMaturity.EXPERIMENTAL
    assert algo_spec(AlgoName.TAIL_SIGNATURE).external_requirements
    assert algo_spec(AlgoName.BOUNDED_TREEWIDTH).maturity is AlgoMaturity.UNAVAILABLE


def test_compiled_problem_and_algo_input_share_one_branch_arithmetic_context():
    problem = parse_input("models/2-colored-graph.wfomcs")
    options = AlgoOptions()
    compiled, _features = compile_reduced_problem(
        begin_reduction(problem),
        options,
    )

    algo_input = build_input(compiled, options=options)

    assert algo_input.arithmetic is compiled.arithmetic


def test_different_branches_may_plan_different_symbol_sets():
    from flint import fmpq_mpoly_ctx

    from wfomc.fol import FOLContext, forall
    from wfomc.problem import Problem
    from wfomc.arithmetic import ArithmeticBackend
    from wfomc.weights import WeightOptions

    fol = FOLContext()
    variable = fol.variable("X")
    predicate = fol.predicate("P", 1)
    sentence = forall(variable, predicate(variable) | ~predicate(variable))
    plain = Problem(sentence=sentence, domain=frozenset((fol.constant("a"),)))
    symbol_context = fmpq_mpoly_ctx.get(("w",), "lex")
    symbolic = Problem(
        sentence=sentence,
        domain=plain.domain,
        weights={predicate: (symbol_context.gen(0), 1)},
    )
    options = AlgoOptions(
        weight_options=WeightOptions(
            precision="round",
            rounded_backend="arb",
        )
    )

    plain_compiled, _ = compile_reduced_problem(begin_reduction(plain), options)
    symbolic_compiled, _ = compile_reduced_problem(
        begin_reduction(symbolic),
        options,
    )

    assert plain_compiled.arithmetic.backend is ArithmeticBackend.ARB
    assert plain_compiled.arithmetic.symbolic_variables == ()
    assert symbolic_compiled.arithmetic.backend is ArithmeticBackend.ARB_POLY
    assert symbolic_compiled.arithmetic.symbolic_variables == ("w",)
    assert plain_compiled.arithmetic is not symbolic_compiled.arithmetic


def test_compile_problem_returns_reduced_problems_and_first_algo_input():
    problem = parse_input("models/unary_evidence/evidence-only.wfomcs")

    artifacts = compile_problem(problem, algo=AlgoName.STANDARD)

    assert artifacts.algo is AlgoName.STANDARD
    assert artifacts.algo_options is not None
    assert artifacts.reduced_problem is not None
    assert len(artifacts.reduced_problem.problems) == 1
    assert len(artifacts.algo_inputs) == 1
    assert artifacts.algo_input is artifacts.algo_inputs[0]
    assert isinstance(artifacts.algo_input, StandardInput)
    assert isinstance(artifacts.reduced_problem.problems[0].problem, ReducedProblem)
    assert artifacts.reduced_problem.problems[0].decoder(4) == 4


@pytest.mark.parametrize(
    ("predicate", "is_circular"),
    (("PRED", False), ("CIRCULAR_PRED", True)),
)
def test_reduced_branch_order_features_drive_incremental_input(
    predicate: str,
    is_circular: bool,
):
    from wfomc.parser import parse_problem

    problem = parse_problem(
        rf"""
\forall X: (\forall Y: ({predicate}(X,Y) | ~{predicate}(X,Y)))
domain = 2
"""
    )

    artifacts = compile_problem(problem, algo=AlgoName.INCREMENTAL)

    assert isinstance(artifacts.algo_input, OrderedCellGraphInput)
    assert artifacts.algo_input.predecessor_orders == (1,)
    assert artifacts.algo_input.has_circular_predecessor is is_circular


def test_compile_problem_materializes_propositional_input_without_decoder():
    problem = parse_input("models/unary_evidence/evidence-only.wfomcs")

    artifacts = compile_problem(problem, algo=AlgoName.PROPOSITIONAL)

    assert isinstance(artifacts.algo_input, GroundCNFInput)
    assert artifacts.algo_input.evidence_unit_clauses


def test_compile_problem_respects_propositional_order_axiom_option():
    problem = parse_input("models/linear_order/head-middle-tail.wfomcs")

    artifacts = compile_problem(
        problem,
        algo=AlgoName.PROPOSITIONAL,
        options=AlgoOptions(linear_order_encoding=LinearOrderEncoding.AXIOMS),
    )

    assert isinstance(artifacts.algo_input, GroundCNFInput)
    assert artifacts.algo_input.linear_order_encoding == LinearOrderEncoding.AXIOMS
    assert artifacts.algo_input.order_unit_clauses


def test_unsupported_features_are_rejected_before_solving():
    problem = parse_input("models/linear_order/head-middle-tail.wfomcs")

    with pytest.raises(UnsupportedFeatureError, match="linear order"):
        compile_problem(problem, algo=AlgoName.STANDARD)

    with pytest.raises(UnsupportedFeatureError, match="linear order"):
        solve(problem, algo=AlgoName.STANDARD)


def test_binary_evidence_is_rejected_instead_of_silently_ignored():
    from dataclasses import replace

    from wfomc.evidence import BinaryEvidence, Evidence, GroundBinaryLiteral
    from wfomc.fol import predicates

    problem = parse_input("models/2-colored-graph.wfomcs")
    predicate = next(pred for pred in predicates(problem.sentence) if pred.arity == 2)
    left, right = sorted(problem.domain, key=str)[:2]
    problem = replace(
        problem,
        evidence=Evidence(
            binary=BinaryEvidence((GroundBinaryLiteral(predicate, left, right, True),))
        ),
    )

    with pytest.raises(UnsupportedFeatureError, match="binary evidence"):
        solve(problem, algo=AlgoName.STANDARD)


def test_compile_problem_delegates_to_algorithm_prepare(monkeypatch):
    from wfomc.algo.core import AlgoSpec, PreparedBranch
    from wfomc.engine import orchestration
    from wfomc.fol import true
    from wfomc.problem import Problem
    from wfomc.reduction import identity_decoder

    problem = parse_input("models/unary_evidence/evidence-only.wfomcs")
    reduced_sentinel = Problem(sentence=true(), domain=frozenset({"a"}))
    algo_input_sentinel = object()
    received = {}

    def fake_prepare(raw_problem, options):
        received.update(problem=raw_problem, options=options)
        return (
            PreparedBranch(reduced_sentinel, algo_input_sentinel, identity_decoder),
        )

    fake_spec = AlgoSpec(
        name=AlgoName.STANDARD,
        resolve_options=lambda features, options=None: AlgoOptions(),
        prepare=fake_prepare,
        solve=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(orchestration, "algo_spec", lambda algo: fake_spec)

    artifacts = compile_problem(problem, algo=AlgoName.STANDARD)

    assert received == {"problem": problem, "options": artifacts.algo_options}
    assert artifacts.algo_input is algo_input_sentinel


def test_compile_problem_rejects_invalid_prepare_contract(monkeypatch):
    from wfomc.algo.core import AlgoSpec
    from wfomc.engine import orchestration

    problem = parse_input("models/unary_evidence/evidence-only.wfomcs")
    fake_spec = AlgoSpec(
        name=AlgoName.STANDARD,
        resolve_options=lambda features, options=None: AlgoOptions(),
        prepare=lambda raw_problem, options: raw_problem,
        solve=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(orchestration, "algo_spec", lambda algo: fake_spec)

    with pytest.raises(TypeError):
        compile_problem(problem, algo=AlgoName.STANDARD)
