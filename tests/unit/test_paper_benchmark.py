from __future__ import annotations

from pathlib import Path

from benchmarks.cases import benchmark_cases
from benchmarks.run_paper import (
    PAPER_BENCHMARK_CASES,
    main,
    paper_benchmark_cases,
)


def test_paper_catalog_is_exactly_the_core_catalog() -> None:
    expected = tuple(case for case in benchmark_cases() if case.category == "core")

    assert paper_benchmark_cases() is PAPER_BENCHMARK_CASES
    assert PAPER_BENCHMARK_CASES == expected
    assert len(PAPER_BENCHMARK_CASES) == 53
    assert all(case.category == "core" for case in PAPER_BENCHMARK_CASES)


def test_paper_runner_uses_a_distinct_default_output(monkeypatch) -> None:
    captured = {}

    def fake_run(cases, args):
        captured["cases"] = cases
        captured["out"] = args.out
        return 0

    monkeypatch.setattr("benchmarks.run_paper.run_catalog", fake_run)

    assert main([]) == 0
    assert captured["cases"] is PAPER_BENCHMARK_CASES
    assert captured["out"] == Path("benchmark-results/paper-core").resolve()
