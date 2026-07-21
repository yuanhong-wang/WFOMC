"""Public engine behavior and domain-separated runtime-cache contracts."""

import gc
import weakref

from wfomc import solve as public_solve
from wfomc.algo import AlgoName
from wfomc.engine import (
    compile_problem,
    instantiate_problem,
    solve as engine_solve,
)
from wfomc.engine.runtime import RuntimeContext, RuntimeOptions
from wfomc.parser import parse_problem_file
from wfomc.problem import Domain
from wfomc.result import WFOMCResult


def test_engine_and_public_entrypoint_return_the_same_result():
    instance = parse_problem_file("models/unary_evidence/evidence-only.wfomcs")

    result = engine_solve(instance, algo=AlgoName.STANDARD)

    assert isinstance(result, WFOMCResult)
    assert result == public_solve(instance, algo=AlgoName.STANDARD)


def test_result_cache_is_scoped_to_one_runtime():
    runtime = RuntimeContext()
    instance = parse_problem_file("models/unary_evidence/evidence-only.wfomcs")

    first = engine_solve(instance, algo=AlgoName.STANDARD, runtime=runtime)
    second = engine_solve(instance, algo=AlgoName.STANDARD, runtime=runtime)

    stats = runtime.cache.stats()
    assert second == first
    assert stats.misses["results"] == 1
    assert stats.hits["results"] == 1


def test_runtime_options_are_accepted_by_engine_and_public_entrypoint():
    instance = parse_problem_file("models/unary_evidence/evidence-only.wfomcs")

    assert engine_solve(
        instance,
        algo=AlgoName.STANDARD,
        runtime=RuntimeOptions(),
    ) == public_solve(instance, algo=AlgoName.STANDARD, runtime=RuntimeOptions())


def test_one_compilation_serves_multiple_domain_sizes():
    runtime = RuntimeContext()
    instance = parse_problem_file("models/2-colored-graph.wfomcs")
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
    instance = parse_problem_file("models/2-colored-graph.wfomcs")
    compiled = compile_problem(instance.problem, algo=AlgoName.FAST, runtime=runtime)

    for size in (1, 2, 3):
        engine_solve(compiled, Domain.of_size(size), runtime=runtime)

    assert runtime.cache.stats().sizes["executions"] == 2


def test_result_cache_is_bounded():
    runtime = RuntimeContext(options=RuntimeOptions(result_cache_size=2))
    instance = parse_problem_file("models/2-colored-graph.wfomcs")
    compiled = compile_problem(instance.problem, algo=AlgoName.FAST, runtime=runtime)

    for size in (1, 2, 3):
        engine_solve(compiled, Domain.of_size(size), runtime=runtime)

    assert runtime.cache.stats().sizes["results"] == 2


def test_input_template_cache_is_bounded():
    runtime = RuntimeContext(options=RuntimeOptions(input_template_cache_size=1))
    instance = parse_problem_file("models/unary_evidence/evidence-only.wfomcs")
    compiled = compile_problem(instance.problem, algo=AlgoName.FAST, runtime=runtime)
    from wfomc.fol import FOLContext

    extended = Domain(
        instance.domain.elements | {FOLContext().constant("cache-extra")}
    )

    engine_solve(compiled, instance.domain, runtime=runtime)
    engine_solve(compiled, extended, runtime=runtime)

    assert runtime.cache.stats().sizes["algo_input_templates"] == 1


def test_evicted_fast_operations_are_released():
    runtime = RuntimeContext(options=RuntimeOptions(execution_cache_size=1))
    instance = parse_problem_file("models/unary_evidence/evidence-only.wfomcs")
    compiled = compile_problem(instance.problem, algo=AlgoName.FAST, runtime=runtime)
    first = instantiate_problem(compiled, instance.domain, runtime=runtime)
    operations = first.algo_input.components[0].weight_operations
    operations_ref = weakref.ref(operations)
    from wfomc.fol import FOLContext

    extended = Domain(
        instance.domain.elements | {FOLContext().constant("cache-release-extra")}
    )

    del operations
    del first
    instantiate_problem(compiled, extended, runtime=runtime)
    gc.collect()

    assert operations_ref() is None
