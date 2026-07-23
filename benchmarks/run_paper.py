#!/usr/bin/env python3
"""Run the Core benchmark catalog used by the paper experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from benchmarks.cases import BenchmarkCase, benchmark_cases
from benchmarks.run import parse_args, run_catalog


ROOT = Path(__file__).resolve().parents[1]
PAPER_BENCHMARK_CASES: tuple[BenchmarkCase, ...] = tuple(
    case for case in benchmark_cases() if case.category == "core"
)


def paper_benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Return the deterministic 53-case paper benchmark catalog."""

    return PAPER_BENCHMARK_CASES


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(
        argv,
        description=__doc__,
        default_out=ROOT / "benchmark-results" / "paper-core",
    )
    if args.worker:
        raise SystemExit("run_paper.py does not expose worker mode")
    return run_catalog(paper_benchmark_cases(), args)


if __name__ == "__main__":
    raise SystemExit(main())
