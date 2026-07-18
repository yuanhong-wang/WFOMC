"""Public engine behavior and domain-separated runtime-cache contracts."""

from wfomc.algo import AlgoName
from wfomc.api import solve as api_solve
from wfomc.engine import compile_problem, solve as engine_solve
from wfomc.engine.runtime import RuntimeContext, RuntimeOptions
from wfomc.parser import parse_input
from wfomc.problem import Domain
from wfomc.result import WFOMCResult


def test_engine_and_public_api_return_the_same_result():
    instance = parse_input("models/unary_evidence/evidence-only.wfomcs")

    result = engine_solve(instance, algo=AlgoName.STANDARD)

    assert isinstance(result, WFOMCResult)
    assert result == api_solve(instance, algo=AlgoName.STANDARD)


def test_result_cache_is_scoped_to_one_runtime():
    runtime = RuntimeContext()
    instance = parse_input("models/unary_evidence/evidence-only.wfomcs")

    first = engine_solve(instance, algo=AlgoName.STANDARD, runtime=runtime)
    second = engine_solve(instance, algo=AlgoName.STANDARD, runtime=runtime)

    stats = runtime.cache.stats()
    assert second == first
    assert stats.misses["results"] == 1
    assert stats.hits["results"] == 1


def test_runtime_options_are_accepted_by_engine_and_api():
    instance = parse_input("models/unary_evidence/evidence-only.wfomcs")

    assert engine_solve(
        instance,
        algo=AlgoName.STANDARD,
        runtime=RuntimeOptions(),
    ) == api_solve(instance, algo=AlgoName.STANDARD, runtime=RuntimeOptions())


def test_one_compilation_serves_multiple_domain_sizes():
    runtime = RuntimeContext()
    instance = parse_input("models/2-colored-graph.wfomcs")
    compiled = compile_problem(
        instance.problem,
        algo=AlgoName.FASTV2,
        runtime=runtime,
    )

    small = engine_solve(compiled, Domain.of_size(2), runtime=runtime)
    large = engine_solve(compiled, Domain.of_size(3), runtime=runtime)

    assert small != large
    stats = runtime.cache.stats()
    assert stats.misses["compiled_problems"] == 1
    assert stats.misses["algo_input_templates"] == 1
    assert stats.hits["algo_input_templates"] == 1
    assert stats.misses["executions"] == 2


def test_concrete_execution_cache_is_bounded():
    runtime = RuntimeContext(options=RuntimeOptions(execution_cache_size=2))
    instance = parse_input("models/2-colored-graph.wfomcs")
    compiled = compile_problem(instance.problem, algo=AlgoName.FAST, runtime=runtime)

    for size in (1, 2, 3):
        engine_solve(compiled, Domain.of_size(size), runtime=runtime)

    assert runtime.cache.stats().sizes["executions"] == 2
