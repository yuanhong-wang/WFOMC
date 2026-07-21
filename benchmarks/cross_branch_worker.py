#!/usr/bin/env python3
"""Run one WFOMC workload using the API installed in this Python environment."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback


SENTINEL = "BENCH_RESULT_JSON="


def _load_api(algorithm: str):
    try:
        from wfomc import AlgoName, parse_problem_file, solve
    except ImportError:
        from wfomc import Algo, parse_input, wfomc
        from loguru import logger

        # The historical branch leaves Loguru's default DEBUG sink installed.
        # Its solver adds its own bounded INFO sink, so remove only the default
        # sink to keep benchmark pipes and timings free of per-state debug logs.
        logger.remove()
        selected = Algo(algorithm)
        return parse_input, lambda problem: wfomc(problem, algo=selected)

    selected = AlgoName(algorithm)
    return parse_problem_file, lambda problem: solve(problem, algo=selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--algorithm", choices=("fastv2", "incremental3"), required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()

    payload: dict[str, object]
    try:
        if args.repetitions < 1:
            raise ValueError("repetitions must be at least one")
        parse_problem_file, solve_problem = _load_api(args.algorithm)
        timings: list[float] = []
        parse_timings: list[float] = []
        results: list[str] = []
        for _ in range(args.repetitions):
            parse_started = time.perf_counter()
            problem = parse_problem_file(args.input)
            parse_timings.append(time.perf_counter() - parse_started)
            started = time.perf_counter()
            result = solve_problem(problem)
            timings.append(time.perf_counter() - started)
            results.append(str(result))
        if len(set(results)) != 1:
            raise RuntimeError("repeated solver invocations returned different results")
        payload = {
            "status": "ok",
            "result": results[-1],
            "solver_time_s": statistics.median(timings),
            "solver_time_min_s": min(timings),
            "solver_time_max_s": max(timings),
            "parse_time_s": statistics.median(parse_timings),
            "repetitions": args.repetitions,
        }
    except BaseException as error:  # benchmark worker must always report diagnostics
        payload = {
            "status": "error",
            "result": None,
            "solver_time_s": None,
            "parse_time_s": None,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=12),
        }

    print(SENTINEL + json.dumps(payload, ensure_ascii=False), flush=True)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
