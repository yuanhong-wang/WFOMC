#!/usr/bin/env python3
"""Canonical current-algorithm benchmark for cold and compile-once protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import psutil

from benchmarks.cases import BenchmarkCase, benchmark_case, benchmark_cases
from benchmarks.cross_branch_performance import (
    MODEL_DOMAIN_SIZES,
    Workload,
    _git_show,
    _kill_process_group,
    _rss_for_tree,
    _tree_models,
    _write_workload,
    collect_catalog_workloads,
    parse_worker_output,
    resolve_commit,
    rewrite_integer_domain,
)


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("boundary-profile", "fastv2", "incremental3")
RESULT_FIELDS = (
    "suite", "protocol", "run_id",
    "series",
    "source_kind", "case",
    "family",
    "category",
    "variant",
    "domain_size",
    "source_sha256", "series_sha256", "comparison_group", "correction_divisor",
    "commit",
    "algorithm",
    "status",
    "compile_time_s", "compile_time_samples_s", "parse_time_s",
    "solver_time_s",
    "solver_time_min_s", "solver_time_max_s", "solver_time_samples_s",
    "series_total_s",
    "wall_time_s",
    "series_peak_rss_bytes",
    "series_peak_rss_mib",
    "result",
    "matches_consensus",
    "comparison_status", "equivalence_status", "is_warm_domain",
    "repetitions",
    "timeout_s",
    "memory_gib",
    "template_hits",
    "template_misses",
    "error",
)
SENTINEL = "BENCH_RESULT_JSON="


@dataclass(frozen=True)
class CaseGroup:
    """Cases that share one domain-free :class:`wfomc.Problem`."""

    series: str
    family: str
    category: str
    variant: str
    cases: tuple[BenchmarkCase, ...]


@dataclass(frozen=True)
class WorkloadGroup:
    """Serialized workloads sharing one exact domain-free problem."""

    series: str
    family: str
    category: str
    variant: str
    workloads: tuple[Workload, ...]


def group_cases(cases: Iterable[BenchmarkCase]) -> tuple[CaseGroup, ...]:
    """Group domain variants only when their domain-free problem keys agree."""

    grouped: dict[tuple[str, str, str, str], list[BenchmarkCase]] = {}
    for case in cases:
        problem_key = repr(case.build_problem().problem.cache_key_parts())
        digest = hashlib.sha256(problem_key.encode()).hexdigest()[:12]
        key = (case.category, case.family, case.variant, digest)
        grouped.setdefault(key, []).append(case)
    result = []
    for (category, family, variant, digest), selected in grouped.items():
        suffix = "" if sum(
            key[:3] == (category, family, variant) for key in grouped
        ) == 1 else f"/{digest}"
        result.append(
            CaseGroup(
                series=f"{category}/{family}/{variant}{suffix}",
                family=family,
                category=category,
                variant=variant,
                cases=tuple(sorted(selected, key=lambda item: (item.domain_size, item.key))),
            )
        )
    return tuple(sorted(result, key=lambda item: item.series))


def group_workloads(workloads: Iterable[Workload]) -> tuple[WorkloadGroup, ...]:
    """Group only inputs whose source differs by the integer domain declaration."""

    grouped: dict[tuple[str, str, str, str, str], list[Workload]] = {}
    for workload in workloads:
        key = (
            workload.source_kind,
            workload.category,
            workload.family,
            workload.variant,
            workload.series_sha256,
        )
        grouped.setdefault(key, []).append(workload)
    result = []
    for (_source_kind, category, family, variant, digest), selected in grouped.items():
        metadata = (category, family, variant)
        collisions = sum(key[1:4] == metadata for key in grouped)
        suffix = "" if collisions == 1 else f"/{digest[:12]}"
        display_family = family.removeprefix(f"{category}/")
        series_name = (
            f"{family}/{variant}{suffix}"
            if category == "model"
            else f"{category}/{display_family}/{variant}{suffix}"
        )
        result.append(
            WorkloadGroup(
                series=series_name,
                family=family,
                category=category,
                variant=variant,
                workloads=tuple(
                    sorted(selected, key=lambda item: (item.domain_size, item.case))
                ),
            )
        )
    return tuple(sorted(result, key=lambda item: item.series))


def build_manifest(
    workloads: Iterable[Workload],
    *,
    protocol: str,
    algorithms: Sequence[str],
    suite: str,
    sources: str,
    commit: str,
    timeout_s: float,
    memory_bytes: int,
    repetitions: int,
    environment: Mapping[str, object],
    dirty_sha256: str,
    lock_sha256: str,
    order_seed: int = 0,
) -> dict[str, object]:
    """Build a deterministic, measurement-complete benchmark manifest."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": protocol,
        "algorithms": list(algorithms),
        "suite": suite,
        "sources": sources,
        "commit": commit,
        "timeout_s": timeout_s,
        "memory_bytes": memory_bytes,
        "repetitions": repetitions,
        "environment": dict(environment),
        "dirty_sha256": dirty_sha256,
        "lock_sha256": lock_sha256,
        "order_seed": order_seed,
        "workloads": [
            {
                "case": workload.case,
                "source_sha256": workload.source_sha256,
                "series_sha256": workload.series_sha256,
                "domain_size": workload.domain_size,
            }
            for workload in sorted(workloads, key=lambda item: item.case)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "run_id": hashlib.sha256(encoded).hexdigest()}


def write_manifest(manifest: Mapping[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_resume(
    output: Path, manifest: Mapping[str, object], *, no_resume: bool
) -> list[dict[str, str]]:
    """Load resumable rows only when their complete manifest identity matches."""

    result_path = output / "results.csv"
    manifest_path = output / "manifest.json"
    if not result_path.exists() or no_resume:
        return []
    if not manifest_path.exists():
        raise ValueError(
            f"existing results at {result_path} have no manifest; use --no-resume "
            "or choose another --out directory"
        )
    saved = json.loads(manifest_path.read_text())
    if saved.get("run_id") != manifest.get("run_id"):
        raise ValueError(
            "existing benchmark manifest does not match this run; use --no-resume "
            "or choose another --out directory"
        )
    with result_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def default_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def working_tree_sha256() -> str:
    """Hash relevant tracked and untracked solver/benchmark changes."""

    digest = hashlib.sha256()
    tracked = subprocess.check_output(
        ("git", "diff", "--binary", "HEAD", "--", "src/wfomc", "benchmarks"),
        cwd=ROOT,
    )
    digest.update(tracked)
    untracked = subprocess.check_output(
        (
            "git", "ls-files", "--others", "--exclude-standard", "--",
            "src/wfomc", "benchmarks",
        ),
        cwd=ROOT,
        text=True,
    )
    for relative in sorted(untracked.splitlines()):
        path = ROOT / relative
        if path.is_file() and "results" not in path.parts:
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def collect_current_model_workloads(
    commit: str,
    domain_sizes: Sequence[int] = MODEL_DOMAIN_SIZES,
) -> tuple[list[Workload], list[dict[str, str]]]:
    """Build the current commit's integer-domain model inventory."""

    workloads: list[Workload] = []
    exclusions: list[dict[str, str]] = []
    for relative in sorted(_tree_models(commit)):
        source = _git_show(commit, relative)
        try:
            rewritten = [rewrite_integer_domain(source, size) for size in domain_sizes]
        except ValueError:
            exclusions.append(
                {"model": relative, "reason": "domain is fixed/named or ambiguous"}
            )
            continue
        for size, concrete_source in zip(domain_sizes, rewritten):
            workloads.append(
                Workload(
                    source_kind="model",
                    case=f"{relative}/n{size}",
                    family=relative,
                    category="model",
                    variant="default",
                    domain_size=size,
                    suffix=Path(relative).suffix,
                    source=concrete_source,
                )
            )
    return workloads, exclusions


def paired_stats(
    rows: Iterable[Mapping[str, object]], comparator: str
) -> dict[str, float | int] | None:
    """Return comparator/BP solve-time ratios on commonly successful cases."""

    grouped: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        if row["status"] == "ok":
            grouped.setdefault(str(row["case"]), {})[str(row["algorithm"])] = row
    ratios = []
    for selected in grouped.values():
        if "boundary-profile" not in selected or comparator not in selected:
            continue
        boundary_time = float(selected["boundary-profile"]["solver_time_s"])
        comparator_time = float(selected[comparator]["solver_time_s"])
        if boundary_time > 0 and comparator_time > 0:
            ratios.append(comparator_time / boundary_time)
    if not ratios:
        return None
    return {
        "pairs": len(ratios),
        "geomean": math.exp(statistics.fmean(math.log(value) for value in ratios)),
        "median": statistics.median(ratios),
        "boundary_faster_pct": 100 * sum(value > 1 for value in ratios) / len(ratios),
    }


class _OperationTimeout(TimeoutError):
    pass


@contextmanager
def _deadline(seconds: float):
    def expire(_signum: int, _frame: object) -> None:
        raise _OperationTimeout(f"operation exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _row_for_case(
    case: BenchmarkCase,
    *,
    status: str,
    solver_time_s: float | None = None,
    result: str | None = None,
    error: str = "",
    solver_time_samples_s: Sequence[float] = (),
    is_warm_domain: bool = False,
) -> dict[str, object]:
    samples = list(solver_time_samples_s)
    return {
        "case": case.key,
        "family": case.family,
        "category": case.category,
        "variant": case.variant,
        "domain_size": case.domain_size,
        "status": status,
        "solver_time_s": solver_time_s,
        "solver_time_min_s": min(samples) if samples else None,
        "solver_time_max_s": max(samples) if samples else None,
        "solver_time_samples_s": samples,
        "is_warm_domain": is_warm_domain,
        "result": result,
        "error": error,
    }


def _exception_status(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    if "Evidence must be consistent with the domain" in message:
        return "invalid"
    if (
        type(error).__name__ == "UnsupportedFeatureError"
        or " does not support " in message
        or " is not supported by " in message
        or " is only supported by " in message
        or "not reducible to UFO2 + cardinality" in message
    ):
        return "unsupported"
    return "error"


def _measure_instances(
    cases: Sequence[BenchmarkCase],
    instances: Sequence[object],
    algorithm: str,
    timeout_s: float,
    repetitions: int,
    protocol: str = "compile-once",
) -> dict[str, object]:
    """Measure already-parsed instances using one selected cache protocol."""

    from wfomc import AlgoName, RuntimeContext, compile_problem, solve

    selected = AlgoName(algorithm)
    if protocol == "cold":
        rows: list[dict[str, object]] = []
        totals = [0.0] * repetitions
        first_stats = None
        for case, instance in zip(cases, instances):
            timings: list[float] = []
            results: list[str] = []
            try:
                for repetition in range(repetitions):
                    runtime = RuntimeContext()
                    started = time.perf_counter()
                    with _deadline(timeout_s):
                        value = solve(instance, algo=selected, runtime=runtime)
                    elapsed = time.perf_counter() - started
                    timings.append(elapsed)
                    totals[repetition] += elapsed
                    results.append(str(value))
                    if first_stats is None:
                        first_stats = runtime.cache.stats()
                if len(set(results)) != 1:
                    raise RuntimeError("repetitions returned different results")
                rows.append(
                    _row_for_case(
                        case, status="ok", solver_time_s=statistics.median(timings),
                        solver_time_samples_s=timings, result=results[0],
                    )
                )
            except _OperationTimeout as error:
                rows.append(_row_for_case(case, status="timeout", error=str(error)))
            except BaseException as error:
                rows.append(
                    _row_for_case(
                        case, status=_exception_status(error),
                        error=f"{type(error).__name__}: {error}",
                    )
                )
        complete = all(row["status"] == "ok" for row in rows)
        return {
            "rows": rows,
            "compile_time_s": None,
            "compile_time_samples_s": [],
            "series_total_s": statistics.median(totals) if complete else None,
            "template_hits": (
                first_stats.hits["algo_input_templates"] if first_stats else 0
            ),
            "template_misses": (
                first_stats.misses["algo_input_templates"] if first_stats else 0
            ),
        }
    if protocol != "compile-once":
        raise ValueError(f"unknown benchmark protocol: {protocol}")

    problem_keys = {repr(instance.problem.cache_key_parts()) for instance in instances}
    if len(problem_keys) != 1:
        raise ValueError("worker cases do not share one domain-free problem")

    compile_times: list[float] = []
    contexts = []
    compiled_problems = []
    try:
        for _ in range(repetitions):
            runtime = RuntimeContext()
            started = time.perf_counter()
            with _deadline(timeout_s):
                compiled = compile_problem(
                    instances[0].problem,
                    algo=selected,
                    runtime=runtime,
                )
            compile_times.append(time.perf_counter() - started)
            contexts.append(runtime)
            compiled_problems.append(compiled)
    except _OperationTimeout as error:
        return {
            "rows": [
                _row_for_case(
                    case,
                    status="timeout" if index == 0 else "skipped-after-resource",
                    error=str(error),
                )
                for index, case in enumerate(cases)
            ],
            "compile_time_s": None,
            "compile_time_samples_s": [],
            "series_total_s": None,
            "template_hits": 0,
            "template_misses": 0,
        }
    except BaseException as error:
        status = _exception_status(error)
        return {
            "rows": [
                _row_for_case(
                    case, status=status,
                    error=f"{type(error).__name__}: {error}",
                )
                for case in cases
            ],
            "compile_time_s": None,
            "compile_time_samples_s": [],
            "series_total_s": None,
            "template_hits": 0,
            "template_misses": 0,
        }

    timings_by_repetition: list[list[float]] = [[] for _ in range(repetitions)]
    rows: list[dict[str, object]] = []
    blocked_status: str | None = None
    blocked_error = ""
    for domain_index, (case, instance) in enumerate(zip(cases, instances)):
        if blocked_status is not None:
            rows.append(
                _row_for_case(
                    case,
                    status=blocked_status,
                    error=blocked_error,
                    is_warm_domain=domain_index > 0,
                )
            )
            continue
        timings: list[float] = []
        results: list[str] = []
        try:
            for repetition, (runtime, compiled) in enumerate(
                zip(contexts, compiled_problems)
            ):
                started = time.perf_counter()
                with _deadline(timeout_s):
                    value = solve(compiled, instance.domain, runtime=runtime)
                elapsed = time.perf_counter() - started
                timings.append(elapsed)
                timings_by_repetition[repetition].append(elapsed)
                results.append(str(value))
            if len(set(results)) != 1:
                raise RuntimeError("repetitions returned different results")
            rows.append(
                _row_for_case(
                    case,
                    status="ok",
                    solver_time_s=statistics.median(timings),
                    solver_time_samples_s=timings,
                    is_warm_domain=domain_index > 0,
                    result=results[0],
                )
            )
        except _OperationTimeout as error:
            rows.append(
                _row_for_case(
                    case, status="timeout", error=str(error),
                    is_warm_domain=domain_index > 0,
                )
            )
            blocked_status = "skipped-after-resource"
            blocked_error = "a smaller domain in this series hit a resource limit"
        except BaseException as error:  # benchmark worker reports solver failures
            status = _exception_status(error)
            message = f"{type(error).__name__}: {error}"
            rows.append(
                _row_for_case(
                    case,
                    status=status,
                    error=message,
                    is_warm_domain=domain_index > 0,
                )
            )
            if status == "unsupported":
                blocked_status = "unsupported"
                blocked_error = message

    stats = contexts[0].cache.stats()
    complete = all(row["status"] == "ok" for row in rows)
    series_total = None
    if complete:
        series_total = statistics.median(
            compile_times[index] + sum(timings_by_repetition[index])
            for index in range(repetitions)
        )
    return {
        "rows": rows,
        "compile_time_s": statistics.median(compile_times),
        "compile_time_samples_s": compile_times,
        "series_total_s": series_total,
        "template_hits": stats.hits["algo_input_templates"],
        "template_misses": stats.misses["algo_input_templates"],
    }


def _run_worker(
    case_keys: Sequence[str],
    algorithm: str,
    timeout_s: float,
    repetitions: int,
    protocol: str = "compile-once",
) -> dict[str, object]:
    """Run catalog cases directly; used by unit tests and worker compatibility."""

    cases = tuple(
        sorted((benchmark_case(key) for key in case_keys), key=lambda case: case.domain_size)
    )
    instances = tuple(case.build_problem() for case in cases)
    return _measure_instances(
        cases, instances, algorithm, timeout_s, repetitions, protocol
    )


@dataclass(frozen=True)
class _ParsedCase:
    key: str
    family: str
    category: str
    variant: str
    domain_size: int


def _run_file_worker(
    paths: Sequence[str],
    algorithm: str,
    timeout_s: float,
    repetitions: int,
    protocol: str,
) -> dict[str, object]:
    from wfomc import parse_problem_file

    parsed = []
    parse_times = []
    for path in paths:
        started = time.perf_counter()
        parsed.append(parse_problem_file(path))
        parse_times.append(time.perf_counter() - started)
    instances = tuple(parsed)
    cases = tuple(
        _ParsedCase(
            key=str(index), family="worker", category="worker", variant="default",
            domain_size=len(instance.domain),
        )
        for index, instance in enumerate(instances)
    )
    payload = _measure_instances(
        cases, instances, algorithm, timeout_s, repetitions, protocol
    )
    for row, parse_time in zip(payload["rows"], parse_times):
        row["parse_time_s"] = parse_time
    return payload


def _worker_main(args: argparse.Namespace) -> int:
    try:
        if args.input:
            payload = _run_file_worker(
                args.input, args.algorithm, args.timeout, args.repetitions,
                args.protocol,
            )
        else:
            payload = _run_worker(
                args.case, args.algorithm, args.timeout, args.repetitions,
                args.protocol,
            )
    except BaseException as error:
        status = _exception_status(error)
        payload = {
            "rows": [
                {
                    "status": status,
                    "solver_time_s": None,
                    "result": None,
                    "error": f"{type(error).__name__}: {error}",
                }
                for _ in (args.input or args.case)
            ],
            "compile_time_s": None,
            "compile_time_samples_s": [],
            "series_total_s": None,
            "template_hits": 0,
            "template_misses": 0,
        }
    print(SENTINEL + json.dumps(payload, ensure_ascii=False), flush=True)
    return 0 if "rows" in payload else 1


def _row_for_workload(
    workload: Workload, *, status: str, error: str
) -> dict[str, object]:
    return {
        "case": workload.case,
        "family": workload.family,
        "category": workload.category,
        "variant": workload.variant,
        "domain_size": workload.domain_size,
        "status": status,
        "solver_time_s": None,
        "result": None,
        "error": error,
    }


def _fallback_rows(
    group: WorkloadGroup, status: str, error: str
) -> list[dict[str, object]]:
    return [
        _row_for_workload(
            workload,
            status=status if index == 0 else "skipped-after-resource",
            error=error,
        )
        for index, workload in enumerate(group.workloads)
    ]


def run_series_process(
    group: WorkloadGroup,
    algorithm: str,
    *,
    protocol: str,
    timeout_s: float,
    memory_bytes: int,
    repetitions: int,
    input_dir: Path,
) -> tuple[dict[str, object], float, int]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--algorithm",
        algorithm,
        "--timeout",
        str(timeout_s),
        "--repetitions",
        str(repetitions),
        "--protocol",
        protocol,
    ]
    for workload in group.workloads:
        command.extend(("--input", str(_write_workload(workload, input_dir))))
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    ps_process = psutil.Process(process.pid)
    peak_rss = 0
    forced_status: str | None = None
    operation_count = len(group.workloads) + (1 if protocol == "compile-once" else 0)
    maximum_wall = timeout_s * repetitions * operation_count + 10
    while process.poll() is None:
        try:
            peak_rss = max(peak_rss, _rss_for_tree(ps_process))
        except psutil.NoSuchProcess:
            pass
        if peak_rss > memory_bytes:
            forced_status = "memory"
            _kill_process_group(process)
            break
        if time.perf_counter() - started > maximum_wall:
            forced_status = "timeout"
            _kill_process_group(process)
            break
        time.sleep(0.02)
    stdout_bytes, stderr_bytes = process.communicate()
    wall_time = time.perf_counter() - started
    stderr = stderr_bytes.decode(errors="replace")[-4000:]
    if forced_status is not None:
        error = (
            f"series exceeded {memory_bytes / 1024**3:g} GiB RSS"
            if forced_status == "memory"
            else f"series exceeded {maximum_wall:g}s wall time"
        )
        return {"rows": _fallback_rows(group, forced_status, error)}, wall_time, peak_rss
    try:
        payload = parse_worker_output(stdout_bytes.decode(errors="replace"))
    except (ValueError, json.JSONDecodeError) as error:
        detail = f"{type(error).__name__}: {error}; stderr={stderr}"
        return {"rows": _fallback_rows(group, "worker-error", detail)}, wall_time, peak_rss
    if "rows" not in payload:
        detail = str(payload.get("error", "worker did not return rows"))
        return {"rows": _fallback_rows(group, "worker-error", detail)}, wall_time, peak_rss
    return payload, wall_time, peak_rss


def mark_correctness(rows: list[dict[str, object]]) -> None:
    """Mark same-input and normalized alternative-encoding correctness."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        row["comparison_status"] = "not-comparable"
        row["equivalence_status"] = (
            "not-comparable" if row.get("comparison_group") else "not-applicable"
        )
        row["matches_consensus"] = None
        grouped.setdefault(str(row["case"]), []).append(row)
    for selected in grouped.values():
        successful = [row for row in selected if row["status"] == "ok"]
        if len(successful) < 2:
            continue
        values = {str(row["result"]) for row in successful}
        status = "match" if len(values) == 1 else "mismatch"
        for row in successful:
            row["comparison_status"] = status
            row["matches_consensus"] = status == "match"

    equivalence: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        group = str(row.get("comparison_group") or "")
        if group and row["status"] == "ok":
            equivalence.setdefault((str(row["algorithm"]), group), []).append(row)
    for selected in equivalence.values():
        if len({str(row["case"]) for row in selected}) < 2:
            continue
        normalized = []
        try:
            for row in selected:
                divisor = int(row.get("correction_divisor") or 1)
                normalized.append(Fraction(str(row["result"])) / divisor)
        except (ValueError, ZeroDivisionError):
            status = "unparseable"
        else:
            status = "match" if len(set(normalized)) == 1 else "mismatch"
        for row in selected:
            row["equivalence_status"] = status


def mark_consensus(rows: list[dict[str, object]]) -> None:
    """Backward-compatible alias for the richer correctness pass."""

    mark_correctness(rows)


def write_csv(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=RESULT_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _format_time(value: object) -> str:
    if value in (None, ""):
        return "-"
    seconds = float(value)
    return f"{seconds * 1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


def _series_paired_stats(
    rows: Iterable[Mapping[str, object]], comparator: str
) -> dict[str, float | int] | None:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        value = row.get("series_total_s")
        if value in (None, ""):
            continue
        grouped.setdefault(str(row["series"]), {})[str(row["algorithm"])] = float(value)
    ratios = [
        selected[comparator] / selected["boundary-profile"]
        for selected in grouped.values()
        if comparator in selected
        and "boundary-profile" in selected
        and selected[comparator] > 0
        and selected["boundary-profile"] > 0
    ]
    if not ratios:
        return None
    return {
        "pairs": len(ratios),
        "geomean": math.exp(statistics.fmean(math.log(value) for value in ratios)),
        "median": statistics.median(ratios),
        "boundary_faster_pct": 100 * sum(value > 1 for value in ratios) / len(ratios),
    }


def write_summary(
    rows: list[dict[str, object]],
    path: Path,
    *,
    suite: str,
    commit: str,
    timeout_s: float,
    memory_gib: float,
    repetitions: int,
    protocol: str = "compile-once",
    algorithms: Sequence[str] = ALGORITHMS,
) -> None:
    lines = [
        f"# Current-algorithm {protocol} performance",
        "",
        f"- Commit: `{commit}`",
        f"- Suite: `{suite}` ({len({row['case'] for row in rows})} workloads)",
        f"- Algorithms: {', '.join(f'`{item}`' for item in algorithms)}",
        f"- Protocol: `{protocol}`",
        f"- Repetitions: {repetitions}; median reported",
        f"- Limits: {timeout_s:g} seconds per compile/solve and {memory_gib:g} GiB RSS per series",
        "",
        "## Status and correctness",
        "",
        "| algorithm | ok | timeout | memory | unsupported | invalid | error | skipped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in algorithms:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        count = lambda status: sum(row["status"] == status for row in selected)
        errors = sum(
            row["status"] not in (
                "ok", "timeout", "memory", "unsupported", "invalid",
                "skipped-after-resource",
            )
            for row in selected
        )
        lines.append(
            f"| {algorithm} | {count('ok')} | {count('timeout')} | "
            f"{count('memory')} | {count('unsupported')} | {count('invalid')} | "
            f"{errors} | {count('skipped-after-resource')} |"
        )
    mismatches = [row for row in rows if row.get("comparison_status") == "mismatch"]
    equivalence_mismatches = [
        row for row in rows if row.get("equivalence_status") == "mismatch"
    ]
    comparable_equivalence_groups = {
        str(row.get("comparison_group"))
        for row in rows
        if row.get("equivalence_status") in ("match", "mismatch")
    }
    successful_by_case: dict[str, int] = {}
    for row in rows:
        if row["status"] == "ok":
            case = str(row["case"])
            successful_by_case[case] = successful_by_case.get(case, 0) + 1
    comparable = {
        case for case, successful_count in successful_by_case.items()
        if successful_count >= 2
    }
    lines.extend(
        [
            "",
            (
                f"All {len(comparable)} workloads with at least two successful algorithms agree."
                if not mismatches
                else f"WARNING: {len({row['case'] for row in mismatches})} workloads disagree."
            ),
            (
                "WARNING: alternative encodings disagree after correction."
                if equivalence_mismatches
                else (
                    f"All {len(comparable_equivalence_groups)} comparable "
                    "alternative-encoding groups agree after correction."
                    if comparable_equivalence_groups
                    else "No alternative-encoding group had enough successful rows "
                    "for comparison."
                )
            ),
            "",
            "## Paired per-domain solve time",
            "",
            "Ratios are `comparator / boundary-profile`; values above 1 favor boundary-profile.",
            "",
            "| comparator | pairs | geometric mean | median | boundary-profile faster |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparator in algorithms[1:]:
        stats = paired_stats(rows, comparator)
        if stats:
            lines.append(
                f"| {comparator} | {stats['pairs']} | {stats['geomean']:.3f}x | "
                f"{stats['median']:.3f}x | {stats['boundary_faster_pct']:.1f}% |"
            )
    if protocol == "compile-once":
        lines.extend(
            [
                "",
                "## First-domain setup versus warm domains",
                "",
                "The first domain includes input-template construction; later domains reuse it.",
                "",
                "| algorithm | first-domain median | warm-domain median |",
                "|---|---:|---:|",
            ]
        )
        for algorithm in algorithms:
            first = [
                float(row["solver_time_s"])
                for row in rows
                if row["algorithm"] == algorithm
                and row["status"] == "ok"
                and str(row.get("is_warm_domain", "False")) == "False"
            ]
            warm = [
                float(row["solver_time_s"])
                for row in rows
                if row["algorithm"] == algorithm
                and row["status"] == "ok"
                and str(row.get("is_warm_domain", "False")) == "True"
            ]
            lines.append(
                f"| {algorithm} | "
                f"{_format_time(statistics.median(first) if first else None)} | "
                f"{_format_time(statistics.median(warm) if warm else None)} |"
            )
    lines.extend(
        [
            "",
            "## Paired end-to-end series time",
            "",
            (
                "Series time includes one compilation plus all increasing-domain solves."
                if protocol == "compile-once"
                else "Cold series time is the total of independently compiled domain solves."
            ),
            "",
            "| comparator | paired series | geometric mean | median | boundary-profile faster |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparator in algorithms[1:]:
        stats = _series_paired_stats(rows, comparator)
        if stats:
            lines.append(
                f"| {comparator} | {stats['pairs']} | {stats['geomean']:.3f}x | "
                f"{stats['median']:.3f}x | {stats['boundary_faster_pct']:.1f}% |"
            )

    lines.extend(
        [
            "",
            "## Per-workload median solve time",
            "",
            "| case | " + " | ".join(algorithms) + " |",
            "|---|" + "---:|" * len(algorithms),
        ]
    )
    by_case: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), {})[str(row["algorithm"])] = row
    for case, selected in sorted(by_case.items()):
        cells = []
        for algorithm in algorithms:
            row = selected.get(algorithm)
            if row is None:
                cells.append("-")
                continue
            cells.append(
                _format_time(row["solver_time_s"])
                if row["status"] == "ok"
                else str(row["status"])
            )
        lines.append(f"| {case} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "## Cache and peak RSS by series",
            "",
            "Template counters come from one repetition; RSS is the peak of the series worker.",
            "",
            "| series / algorithm | template misses | template hits | peak RSS | total time |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["series"]), str(row["algorithm"]))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {key[0]} / {key[1]} | {row.get('template_misses', '-')} | "
            f"{row.get('template_hits', '-')} | "
            f"{float(row['series_peak_rss_mib']):.1f} MiB | "
            f"{_format_time(row.get('series_total_s'))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="core-main")
    parser.add_argument(
        "--protocol", choices=("cold", "compile-once"), default="compile-once"
    )
    parser.add_argument(
        "--sources", choices=("all", "catalog", "models"), default="catalog"
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS)
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--memory-gib", type=float, default=4.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--limit-workloads", type=int, default=None)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "benchmarks/results/current_algorithms",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=ALGORITHMS, help=argparse.SUPPRESS)
    parser.add_argument("--case", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--input", action="append", default=[], help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0 or args.memory_gib <= 0 or args.repetitions < 1:
        raise SystemExit("timeout, memory, and repetitions must be positive")
    if args.worker:
        if args.algorithm is None or not (args.case or args.input):
            raise SystemExit("worker mode requires --algorithm and --case/--input")
        return _worker_main(args)

    commit = resolve_commit("HEAD")
    workloads: list[Workload] = []
    exclusions: list[dict[str, str]] = []
    if args.sources in ("all", "catalog"):
        workloads.extend(collect_catalog_workloads(args.suite))
    if args.sources in ("all", "models"):
        model_workloads, exclusions = collect_current_model_workloads(commit)
        workloads.extend(model_workloads)
    if args.limit_workloads is not None:
        workloads = workloads[: args.limit_workloads]
    if not workloads:
        raise SystemExit("selected benchmark inventory is empty")
    if args.protocol == "compile-once":
        groups = group_workloads(workloads)
    else:
        groups = tuple(
            WorkloadGroup(
                series=(
                    (
                        f"{workload.family}/{workload.variant}"
                        if workload.category == "model"
                        else f"{workload.category}/"
                        f"{workload.family.removeprefix(f'{workload.category}/')}/"
                        f"{workload.variant}"
                    )
                    + f"/n{workload.domain_size}"
                ),
                family=workload.family,
                category=workload.category,
                variant=workload.variant,
                workloads=(workload,),
            )
            for workload in workloads
        )
    args.out.mkdir(parents=True, exist_ok=True)
    input_dir = args.out / ".inputs"
    input_dir.mkdir(exist_ok=True)
    result_path = args.out / "results.csv"
    memory_bytes = int(args.memory_gib * 1024**3)
    manifest = build_manifest(
        workloads,
        protocol=args.protocol,
        algorithms=tuple(args.algorithms),
        suite=args.suite,
        sources=args.sources,
        commit=commit,
        timeout_s=args.timeout,
        memory_bytes=memory_bytes,
        repetitions=args.repetitions,
        environment=default_environment(),
        dirty_sha256=working_tree_sha256(),
        lock_sha256=_file_sha256(ROOT / "uv.lock"),
        order_seed=args.order_seed,
    )
    try:
        resume_rows = prepare_resume(args.out, manifest, no_resume=args.no_resume)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    write_manifest(manifest, args.out / "manifest.json")
    if exclusions:
        with (args.out / "excluded.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("model", "reason"))
            writer.writeheader()
            writer.writerows(exclusions)

    resume_by_key = {
        (str(row["case"]), str(row["algorithm"])): row for row in resume_rows
        if row.get("run_id") == manifest["run_id"]
    }
    rows: list[dict[str, object]] = []
    tasks = [(group, algorithm) for group in groups for algorithm in args.algorithms]
    random.Random(args.order_seed).shuffle(tasks)
    total = len(tasks)
    for index, (group, algorithm) in enumerate(tasks, 1):
        previous = [
            resume_by_key.get((workload.case, algorithm))
            for workload in group.workloads
        ]
        if all(row is not None for row in previous):
            rows.extend(dict(row) for row in previous if row is not None)
        else:
            print(f"[{index}/{total}] {algorithm} {group.series}", flush=True)
            payload, wall_time, peak_rss = run_series_process(
                group,
                algorithm,
                protocol=args.protocol,
                timeout_s=args.timeout,
                memory_bytes=memory_bytes,
                repetitions=args.repetitions,
                input_dir=input_dir,
            )
            for workload, measured in zip(group.workloads, payload["rows"]):
                rows.append(
                    {
                        "suite": args.suite,
                        "protocol": args.protocol,
                        "run_id": manifest["run_id"],
                        "series": group.series,
                        **measured,
                        "source_kind": workload.source_kind,
                        "case": workload.case,
                        "family": workload.family,
                        "category": workload.category,
                        "variant": workload.variant,
                        "domain_size": workload.domain_size,
                        "source_sha256": workload.source_sha256,
                        "series_sha256": workload.series_sha256,
                        "comparison_group": workload.comparison_group,
                        "correction_divisor": workload.correction_divisor,
                        "commit": commit,
                        "algorithm": algorithm,
                        "compile_time_s": payload.get("compile_time_s"),
                        "compile_time_samples_s": payload.get(
                            "compile_time_samples_s", []
                        ),
                        "series_total_s": payload.get("series_total_s"),
                        "wall_time_s": wall_time,
                        "series_peak_rss_bytes": peak_rss,
                        "series_peak_rss_mib": peak_rss / 1024**2,
                        "matches_consensus": None,
                        "repetitions": args.repetitions,
                        "timeout_s": args.timeout,
                        "memory_gib": args.memory_gib,
                        "template_hits": payload.get("template_hits", ""),
                        "template_misses": payload.get("template_misses", ""),
                    }
                )
            mark_consensus(rows)
            write_csv(rows, result_path)
    mark_consensus(rows)
    write_csv(rows, result_path)
    write_summary(
        rows,
        args.out / "summary.md",
        suite=args.suite,
        commit=commit,
        timeout_s=args.timeout,
        memory_gib=args.memory_gib,
        repetitions=args.repetitions,
        protocol=args.protocol,
        algorithms=tuple(args.algorithms),
    )
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
