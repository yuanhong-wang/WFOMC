#!/usr/bin/env python3
"""Run the fixed Core domain grids used by the paper experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from benchmarks.cases import BenchmarkCase, core_case
from benchmarks.run import parse_args, run_catalog, worker_main


ROOT = Path(__file__).resolve().parents[1]
PAPER_DOMAIN_GRIDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    (
        "permutations",
        (20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 125),
    ),
    (
        "undirected-2-regular",
        (20, 40, 60, 80, 100, 120, 140, 160, 180, 200),
    ),
    (
        "undirected-3-regular",
        (20, 30, 40, 50, 60, 70, 80, 90, 100),
    ),
    (
        "undirected-4-regular",
        (12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50, 52, 56, 60),
    ),
    (
        "properly-2-coloured-graph",
        (100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000),
    ),
    (
        "properly-3-coloured-graph",
        (50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300),
    ),
    (
        "properly-4-coloured-graph",
        (20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150),
    ),
    (
        "properly-5-coloured-graph",
        (16, 24, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75),
    ),
    (
        "derangements",
        (50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300),
    ),
    (
        "endofunctions",
        (40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240),
    ),
    (
        "loopless-digraph-without-isolates",
        (50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600),
    ),
    (
        "2-edge-disjoint-perfect-matchings",
        (20, 40, 60, 80, 100, 150, 200, 250, 300, 400),
    ),
    (
        "3-edge-disjoint-perfect-matchings",
        (10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
    ),
    (
        "4-edge-disjoint-perfect-matchings",
        (6, 8, 10, 12, 14, 16),
    ),
)

PAPER_BENCHMARK_CASES: tuple[BenchmarkCase, ...] = tuple(
    core_case(family, domain_size)
    for family, domain_sizes in PAPER_DOMAIN_GRIDS
    for domain_size in domain_sizes
)
_PAPER_CASES_BY_KEY = {case.key: case for case in PAPER_BENCHMARK_CASES}


def paper_benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Return the deterministic fixed-grid paper benchmark catalog."""

    return PAPER_BENCHMARK_CASES


def paper_benchmark_case(key: str) -> BenchmarkCase:
    """Look up one concrete paper case by its stable key."""

    try:
        return _PAPER_CASES_BY_KEY[key]
    except KeyError as error:
        raise KeyError(f"unknown Core paper case: {key}") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(
        argv,
        description=__doc__,
        default_out=ROOT / "benchmark-results" / "paper-core",
        default_protocol="cold-stop",
        default_timeout=300.0,
        default_repetitions=3,
    )
    if args.worker:
        if args.algorithm is None or not args.case:
            raise SystemExit("worker mode requires --algorithm and --case")
        return worker_main(args, case_lookup=paper_benchmark_case)
    if args.protocol != "cold-stop":
        raise SystemExit("the paper benchmark requires --protocol cold-stop")
    return run_catalog(
        paper_benchmark_cases(),
        args,
        worker_module="benchmarks.run_paper",
    )


if __name__ == "__main__":
    raise SystemExit(main())
