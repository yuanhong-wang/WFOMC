"""Public engine behavior and runtime-cache contracts."""

from fractions import Fraction

from wfomc.algo import AlgoName
from wfomc.api import solve as api_solve
from wfomc.engine import compile_problem, solve as engine_solve
from wfomc.engine.runtime import RuntimeContext, RuntimeOptions
from wfomc.fol import Constant, Predicate, Variable, atom, forall
from wfomc.parser import parse_input
from wfomc.problem import Problem
from wfomc.result import WFOMCResult


def test_engine_and_public_api_return_the_same_result():
    problem = parse_input("models/unary_evidence/evidence-only.wfomcs")

    result = engine_solve(problem, algo=AlgoName.STANDARD)

    assert isinstance(result, WFOMCResult)
    assert result == api_solve(problem, algo=AlgoName.STANDARD)


def test_result_cache_is_scoped_to_one_runtime():
    runtime = RuntimeContext()
    problem = parse_input("models/unary_evidence/evidence-only.wfomcs")

    first = engine_solve(problem, algo=AlgoName.STANDARD, runtime=runtime)
    second = engine_solve(problem, algo=AlgoName.STANDARD, runtime=runtime)

    stats = runtime.cache.stats()
    assert second == first
    assert stats.misses["results"] == 1
    assert stats.hits["results"] == 1


def test_runtime_options_are_accepted_by_engine_and_api():
    problem = parse_input("models/unary_evidence/evidence-only.wfomcs")

    assert engine_solve(
        problem,
        algo=AlgoName.STANDARD,
        runtime=RuntimeOptions(),
    ) == api_solve(problem, algo=AlgoName.STANDARD, runtime=RuntimeOptions())


def test_compile_cache_reuses_features_and_prepared_input():
    runtime = RuntimeContext()
    problem = parse_input("models/unary_evidence/evidence-only.wfomcs")

    compile_problem(problem, algo=AlgoName.STANDARD, runtime=runtime)
    compile_problem(problem, algo=AlgoName.STANDARD, runtime=runtime)

    stats = runtime.cache.stats()
    assert (stats.misses["features"], stats.hits["features"]) == (1, 1)
    assert (stats.misses["algo_inputs"], stats.hits["algo_inputs"]) == (1, 1)


def test_compile_cache_key_includes_domain():
    runtime = RuntimeContext()
    x = Variable("X")
    predicate = Predicate("P", 1)
    sentence = forall(x, atom("P", x))
    weights = {predicate: (Fraction(2, 1), Fraction(3, 1))}
    small = Problem(
        sentence=sentence,
        domain=frozenset({Constant("a"), Constant("b")}),
        weights=weights,
    )
    large = Problem(
        sentence=sentence,
        domain=frozenset({Constant("a"), Constant("b"), Constant("c")}),
        weights=weights,
    )

    compile_problem(small, algo=AlgoName.STANDARD, runtime=runtime)
    compile_problem(large, algo=AlgoName.STANDARD, runtime=runtime)

    stats = runtime.cache.stats()
    assert stats.misses["features"] == 2
    assert stats.misses["algo_inputs"] == 2
