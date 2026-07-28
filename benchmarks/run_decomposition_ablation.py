#!/usr/bin/env python3
"""Measure Boundary-Profile decomposition strategies on typed graphs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import psutil

from benchmarks.run import resolve_commit
from benchmarks.run_paper import build_paper_benchmark_case


ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "DECOMPOSITION_ABLATION_JSON="
FAMILIES = (
    "typed-path-relation-k8",
    "typed-tree-relation-k8",
    "typed-cycle-relation-k8",
    "typed-asymmetric-relation-k8",
)
STRATEGIES = (
    "heuristic-only",
    "exact-subset-cost",
    "tail-caterpillar",
    "greedy-agglomerative",
)
FIELDS = (
    "family",
    "domain_size",
    "requested_strategy",
    "selected_strategy",
    "status",
    "runtime_s",
    "runtime_samples_s",
    "bp_width",
    "join_width",
    "estimated_states",
    "estimated_join_pairs",
    "estimated_log_work",
    "peak_rss_mib",
    "result_sha256",
    "repetitions",
    "timeout_s",
    "memory_gib",
    "commit",
    "error",
)


def _worker(
    family: str,
    strategy: str,
    domain_size: int,
    repetitions: int,
) -> dict[str, object]:
    from wfomc import (
        AlgoName,
        AlgoOptions,
        BoundaryProfileOptions,
        RuntimeContext,
        compile_problem,
        instantiate_problem,
        solve,
    )

    case = build_paper_benchmark_case(family, domain_size)
    instance = case.build_problem()
    options = AlgoOptions(
        boundary_profile_options=BoundaryProfileOptions(
            planner_strategy=strategy,
        )
    )
    runtimes = []
    result_hashes = []
    selected_plan = None
    for _ in range(repetitions):
        runtime = RuntimeContext()
        started = time.perf_counter()
        compiled = compile_problem(
            instance.problem,
            algo=AlgoName.BOUNDARY_PROFILE,
            options=options,
            runtime=runtime,
        )
        value = solve(compiled, instance.domain, runtime=runtime)
        runtimes.append(time.perf_counter() - started)
        result_hashes.append(hashlib.sha256(str(value).encode()).hexdigest())
        execution = instantiate_problem(
            compiled,
            instance.domain,
            runtime=runtime,
        )
        plans = [
            component.plan
            for branch in execution.branches
            for component in branch.algo_input.components
        ]
        if len(plans) != 1 or plans[0] is None:
            raise RuntimeError(
                f"expected one Boundary-Profile plan, found {len(plans)}"
            )
        if selected_plan is not None and plans[0] != selected_plan:
            raise RuntimeError("repetitions selected different plans")
        selected_plan = plans[0]
    if len(set(result_hashes)) != 1:
        raise RuntimeError("repetitions returned different results")
    assert selected_plan is not None
    return {
        "family": family,
        "domain_size": domain_size,
        "requested_strategy": strategy,
        "selected_strategy": selected_plan.strategy,
        "status": "ok",
        "runtime_s": statistics.median(runtimes),
        "runtime_samples_s": runtimes,
        "bp_width": selected_plan.bp_width,
        "join_width": selected_plan.join_width,
        "estimated_states": selected_plan.estimated_states,
        "estimated_join_pairs": selected_plan.estimated_join_pairs,
        "estimated_log_work": selected_plan.estimated_log_work,
        "result_sha256": result_hashes[0],
        "error": "",
    }


def _rss_for_tree(process: psutil.Process) -> int:
    total = 0
    try:
        children = process.children(recursive=True)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        children = []
    for item in (process, *children):
        try:
            total += item.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        process.kill()


def _run_process(
    *,
    family: str,
    strategy: str,
    domain_size: int,
    repetitions: int,
    timeout_s: float,
    memory_bytes: int,
) -> dict[str, object]:
    command = (
        sys.executable,
        "-m",
        "benchmarks.run_decomposition_ablation",
        "--worker",
        "--family",
        family,
        "--strategy",
        strategy,
        "--domain-size",
        str(domain_size),
        "--repetitions",
        str(repetitions),
    )
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    monitored = psutil.Process(process.pid)
    started = time.monotonic()
    peak_rss = 0
    forced_status = ""
    while process.poll() is None:
        peak_rss = max(peak_rss, _rss_for_tree(monitored))
        if peak_rss > memory_bytes:
            forced_status = "memory"
            _kill_process_group(process)
            break
        if time.monotonic() - started > timeout_s:
            forced_status = "timeout"
            _kill_process_group(process)
            break
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    peak_rss = max(peak_rss, _rss_for_tree(monitored))
    if forced_status:
        return {
            "family": family,
            "domain_size": domain_size,
            "requested_strategy": strategy,
            "selected_strategy": "",
            "status": forced_status,
            "runtime_s": "",
            "runtime_samples_s": [],
            "bp_width": "",
            "join_width": "",
            "estimated_states": "",
            "estimated_join_pairs": "",
            "estimated_log_work": "",
            "result_sha256": "",
            "error": (
                f"worker exceeded {timeout_s:g} seconds"
                if forced_status == "timeout"
                else f"worker exceeded {memory_bytes / 1024**3:g} GiB RSS"
            ),
            "peak_rss_mib": peak_rss / 1024**2,
        }
    output = stdout.decode(errors="replace")
    payload = None
    for line in reversed(output.splitlines()):
        if line.startswith(SENTINEL):
            payload = json.loads(line[len(SENTINEL) :])
            break
    if process.returncode or not isinstance(payload, dict):
        return {
            "family": family,
            "domain_size": domain_size,
            "requested_strategy": strategy,
            "selected_strategy": "",
            "status": "error",
            "runtime_s": "",
            "runtime_samples_s": [],
            "bp_width": "",
            "join_width": "",
            "estimated_states": "",
            "estimated_join_pairs": "",
            "estimated_log_work": "",
            "result_sha256": "",
            "error": stderr.decode(errors="replace")[-4000:],
            "peak_rss_mib": peak_rss / 1024**2,
        }
    return {**payload, "peak_rss_mib": peak_rss / 1024**2}


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row.get(field))
                        if isinstance(row.get(field), (list, tuple))
                        else row.get(field, "")
                    )
                    for field in FIELDS
                }
            )


def _ratio(
    by_strategy: Mapping[str, Mapping[str, object]],
    numerator: str,
    denominator: str,
) -> str:
    upper = by_strategy.get(numerator, {})
    lower = by_strategy.get(denominator, {})
    if upper.get("status") != "ok" or lower.get("status") != "ok":
        return f"{upper.get('status', '-')}/{lower.get('status', '-')}"
    return f"{float(upper['runtime_s']) / float(lower['runtime_s']):.3f}"


def _write_summary(
    rows: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    labels = {
        "typed-path-relation-k8": "Path",
        "typed-tree-relation-k8": "Tree",
        "typed-cycle-relation-k8": "Cycle",
        "typed-asymmetric-relation-k8": "Asymmetric",
    }
    lines = [
        "# Typed decomposition ablation",
        "",
        "| graph | heuristic selected | JBPW H/E | H/E runtime | Tail/H | Greedy/H |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        by_strategy = {
            str(row["requested_strategy"]): row
            for row in rows
            if row["family"] == family
        }
        heuristic = by_strategy.get("heuristic-only", {})
        exact = by_strategy.get("exact-subset-cost", {})
        width = (
            f"{heuristic.get('join_width', '-')}/{exact.get('join_width', '-')}"
        )
        lines.append(
            f"| {labels[family]} | {heuristic.get('selected_strategy', '-')} | "
            f"{width} | "
            f"{_ratio(by_strategy, 'heuristic-only', 'exact-subset-cost')} | "
            f"{_ratio(by_strategy, 'tail-caterpillar', 'heuristic-only')} | "
            f"{_ratio(by_strategy, 'greedy-agglomerative', 'heuristic-only')} |"
        )
    hashes_by_family = {
        family: {
            str(row["result_sha256"])
            for row in rows
            if row["family"] == family and row.get("status") == "ok"
        }
        for family in FAMILIES
    }
    all_agree = all(
        hashes and len(hashes) == 1 for hashes in hashes_by_family.values()
    )
    lines.extend(
        [
            "",
            (
                "All successful strategies agree exactly."
                if all_agree
                else "Result hashes require per-family inspection."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark-results")
    parser.add_argument("--domain-size", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--memory-gib", type=float, default=4.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--family", choices=FAMILIES, help=argparse.SUPPRESS)
    parser.add_argument("--strategy", choices=STRATEGIES, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.domain_size <= 0:
        raise SystemExit("domain size must be positive")
    if args.repetitions <= 0 or args.timeout <= 0 or args.memory_gib <= 0:
        raise SystemExit("repetitions and resource limits must be positive")
    if args.worker:
        if args.family is None or args.strategy is None:
            raise SystemExit("worker mode requires family and strategy")
        try:
            payload = _worker(
                args.family,
                args.strategy,
                args.domain_size,
                args.repetitions,
            )
        except BaseException as error:
            payload = {
                "family": args.family,
                "domain_size": args.domain_size,
                "requested_strategy": args.strategy,
                "selected_strategy": "",
                "status": "error",
                "runtime_s": "",
                "runtime_samples_s": [],
                "bp_width": "",
                "join_width": "",
                "estimated_states": "",
                "estimated_join_pairs": "",
                "estimated_log_work": "",
                "result_sha256": "",
                "error": f"{type(error).__name__}: {error}",
            }
        print(SENTINEL + json.dumps(payload, sort_keys=True))
        return 0
    commit = resolve_commit()
    memory_bytes = int(args.memory_gib * 1024**3)
    rows = []
    for family in FAMILIES:
        for strategy in STRATEGIES:
            print(f"{family}: {strategy}", flush=True)
            measured = _run_process(
                family=family,
                strategy=strategy,
                domain_size=args.domain_size,
                repetitions=args.repetitions,
                timeout_s=args.timeout,
                memory_bytes=memory_bytes,
            )
            rows.append(
                {
                    **measured,
                    "repetitions": args.repetitions,
                    "timeout_s": args.timeout,
                    "memory_gib": args.memory_gib,
                    "commit": commit,
                }
            )
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.out / "results.csv")
    _write_summary(rows, args.out / "summary.md")
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "domain_size": args.domain_size,
                "families": FAMILIES,
                "strategies": STRATEGIES,
                "timeout_s": args.timeout,
                "memory_gib": args.memory_gib,
                "repetitions": args.repetitions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
