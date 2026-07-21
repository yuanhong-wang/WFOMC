"""End-to-end cache reuse across a growing domain-size series."""

from __future__ import annotations

from wfomc import (
    AlgoName,
    Domain,
    RuntimeContext,
    compile_problem,
    instantiate_problem,
    parse_problem_file,
    solve,
)


def test_compile_once_domain_series_reuses_each_cache_at_its_own_lifetime():
    problem = parse_problem_file("models/2-colored-graph.wfomcs").problem
    runtime = RuntimeContext()
    compiled = compile_problem(
        problem,
        algo=AlgoName.FASTV2,
        runtime=runtime,
    )

    assert (
        compile_problem(
            problem,
            algo=AlgoName.FASTV2,
            runtime=runtime,
        )
        is compiled
    )

    domains = tuple(Domain.of_size(size) for size in (1, 2, 3, 4))
    results = []
    for domain in domains:
        execution = instantiate_problem(compiled, domain, runtime=runtime)
        assert instantiate_problem(compiled, domain, runtime=runtime) is execution
        results.append(int(solve(compiled, domain, runtime=runtime)))

    for domain in domains:
        solve(compiled, domain, runtime=runtime)

    assert results == [2, 6, 26, 162]
    stats = runtime.cache.stats()
    assert stats.misses == {
        "features": 1,
        "compiled_problems": 1,
        "algo_input_templates": 1,
        "executions": 4,
        "results": 4,
    }
    assert stats.hits == {
        "features": 1,
        "compiled_problems": 1,
        "algo_input_templates": 3,
        "executions": 8,
        "results": 4,
    }
    assert stats.sizes == {
        "features": 1,
        "compiled_problems": 1,
        "algo_input_templates": 1,
        "executions": 4,
        "results": 4,
    }
