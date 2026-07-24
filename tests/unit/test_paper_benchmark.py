from __future__ import annotations

from pathlib import Path

from benchmarks.run_paper import (
    PAPER_BENCHMARK_CASES,
    PAPER_DOMAIN_GRIDS,
    main,
    paper_benchmark_case,
    paper_benchmark_cases,
)


EXPECTED_DOMAINS = {
    "permutations": (20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 125),
    "undirected-2-regular": (20, 40, 60, 80, 100, 120, 140, 160, 180, 200),
    "undirected-3-regular": (20, 30, 40, 50, 60, 70, 80, 90, 100),
    "undirected-4-regular": (
        12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50, 52, 56, 60,
    ),
    "properly-2-coloured-graph": (
        100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000,
    ),
    "properly-3-coloured-graph": (
        50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300,
    ),
    "properly-4-coloured-graph": (
        20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150,
    ),
    "properly-5-coloured-graph": (
        16, 24, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
    ),
    "derangements": (50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300),
    "endofunctions": (40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240),
    "loopless-digraph-without-isolates": (
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
    ),
    "2-edge-disjoint-perfect-matchings": (
        20, 40, 60, 80, 100, 150, 200, 250, 300, 400,
    ),
    "3-edge-disjoint-perfect-matchings": (
        10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    ),
    "4-edge-disjoint-perfect-matchings": (6, 8, 10, 12, 14, 16),
}


def test_paper_catalog_uses_the_fixed_domain_grids() -> None:
    assert paper_benchmark_cases() is PAPER_BENCHMARK_CASES
    assert len(PAPER_DOMAIN_GRIDS) == 14
    assert len(PAPER_BENCHMARK_CASES) == 151
    assert all(case.category == "core" for case in PAPER_BENCHMARK_CASES)
    actual = {
        family: tuple(
            case.domain_size
            for case in PAPER_BENCHMARK_CASES
            if case.family == family
        )
        for family in EXPECTED_DOMAINS
    }
    assert actual == EXPECTED_DOMAINS
    assert len({case.key for case in PAPER_BENCHMARK_CASES}) == 151


def test_paper_case_lookup_supports_fixed_grid_only() -> None:
    case = paper_benchmark_case(
        "core/properly-2-coloured-graph/n2000"
    )
    assert case.domain_size == 2000

    try:
        paper_benchmark_case("core/properly-2-coloured-graph/n2001")
    except KeyError as error:
        assert "unknown Core paper case" in str(error)
    else:
        raise AssertionError("out-of-grid paper case should not be found")


def test_paper_runner_uses_formal_experiment_defaults(monkeypatch) -> None:
    captured = {}

    def fake_run(cases, args, *, worker_module):
        captured["cases"] = cases
        captured["out"] = args.out
        captured["protocol"] = args.protocol
        captured["timeout"] = args.timeout
        captured["repetitions"] = args.repetitions
        captured["worker_module"] = worker_module
        return 0

    monkeypatch.setattr("benchmarks.run_paper.run_catalog", fake_run)

    assert main([]) == 0
    assert captured["cases"] is PAPER_BENCHMARK_CASES
    assert captured["out"] == Path("benchmark-results/paper-core").resolve()
    assert captured["protocol"] == "cold-stop"
    assert captured["timeout"] == 300.0
    assert captured["repetitions"] == 3
    assert captured["worker_module"] == "benchmarks.run_paper"
