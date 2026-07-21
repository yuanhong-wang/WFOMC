#!/usr/bin/env python3
"""Reproduce the historical BP comparison against saved cross-branch results."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping

from benchmarks.cross_branch_performance import (
    RESOURCE_FAILURES,
    RESULT_FIELDS,
    Workload,
    _resume_identity_matches,
    _skipped_row,
    _write_workload,
    benchmark_run_id,
    collect_catalog_workloads,
    read_csv,
    resolve_commit,
    rewrite_integer_domain,
    run_worker,
    series_key,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = "boundary-profile"
BRANCH = "current"


def _git_show(commit: str, relative_path: str) -> str:
    return subprocess.check_output(
        ("git", "show", f"{commit}:{relative_path}"), cwd=ROOT, text=True
    )


def reconstruct_workloads(
    saved_rows: Iterable[Mapping[str, object]],
) -> tuple[list[Workload], str]:
    """Rebuild the exact inputs represented by a saved cross-branch batch."""

    rows = list(saved_rows)
    baseline = [
        row for row in rows
        if row["branch"] == "devel" and row["algorithm"] == "fastv2"
    ]
    if not baseline:
        raise ValueError("saved results contain no devel/fastv2 workload inventory")
    commits = {str(row["commit"]) for row in baseline}
    if len(commits) != 1:
        raise ValueError(f"saved devel inventory has multiple commits: {sorted(commits)}")
    source_commit = commits.pop()

    catalog = {workload.case: workload for workload in collect_catalog_workloads("all")}
    model_sources: dict[str, str] = {}
    workloads: list[Workload] = []
    seen: set[tuple[str, int]] = set()
    for row in baseline:
        key = (str(row["source_sha256"]), int(row["domain_size"]))
        if key in seen:
            continue
        seen.add(key)
        source_kind = str(row["source_kind"])
        case = str(row["case"])
        if source_kind == "catalog":
            try:
                workload = catalog[case]
            except KeyError as error:
                raise ValueError(f"saved catalog case no longer exists: {case}") from error
        elif source_kind == "model":
            family = str(row["family"])
            if family not in model_sources:
                model_sources[family] = _git_show(source_commit, family)
            original = model_sources[family]
            domain_size = int(row["domain_size"])
            workload = Workload(
                source_kind="model",
                case=case,
                family=family,
                category=str(row["category"]),
                variant=str(row["variant"]),
                domain_size=domain_size,
                suffix=Path(family).suffix,
                source=rewrite_integer_domain(original, domain_size),
            )
        else:
            raise ValueError(f"unknown source kind in saved results: {source_kind}")
        if workload.source_sha256 != key[0]:
            raise ValueError(
                f"source hash mismatch for {case}: saved {key[0]}, "
                f"reconstructed {workload.source_sha256}"
            )
        workloads.append(workload)
    if len(workloads) != len(seen):
        raise AssertionError("workload reconstruction lost rows")
    return workloads, source_commit


def execute_boundary_profile(
    workloads: Iterable[Workload],
    *,
    python: Path,
    commit: str,
    timeout_s: float,
    memory_bytes: int,
    repetitions: int,
    input_dir: Path,
    checkpoint_path: Path | None = None,
    resume_rows: Iterable[Mapping[str, object]] = (),
    run_id: str = "",
) -> list[dict[str, object]]:
    """Run one configuration, truncating a series after a resource failure."""

    ordered = sorted(
        workloads,
        key=lambda item: (
            item.source_kind, item.family, item.variant, item.domain_size, item.case
        ),
    )
    resumable = {
        (str(row["source_sha256"]), int(row["domain_size"])): dict(row)
        for row in resume_rows
        if row.get("commit") == commit
        and row.get("algorithm") == ALGORITHM
        and row.get("status") != "skipped-after-resource"
    }
    rows: list[dict[str, object]] = []
    blocked: set[tuple[object, ...]] = set()
    for index, workload in enumerate(ordered, 1):
        metadata = {
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
            "branch": BRANCH,
            "commit": commit,
            "algorithm": ALGORITHM,
        }
        workload_series = series_key(metadata)
        if workload_series in blocked:
            row = _skipped_row(
                workload, BRANCH, commit, ALGORITHM, repetitions,
                timeout_s=timeout_s, memory_bytes=memory_bytes, run_id=run_id,
            )
        else:
            candidate = resumable.get((workload.source_sha256, workload.domain_size))
            previous = (
                candidate
                if candidate is not None
                and _resume_identity_matches(
                    candidate, commit=commit, repetitions=repetitions,
                    timeout_s=timeout_s, memory_bytes=memory_bytes,
                    run_id=run_id,
                )
                else None
            )
            if previous is not None:
                row = {**previous, **metadata, "matches_consensus": None}
            else:
                print(f"[{index}/{len(ordered)}] {ALGORITHM} {workload.case}", flush=True)
                path = _write_workload(workload, input_dir)
                measured = run_worker(
                    python,
                    path,
                    ALGORITHM,
                    timeout_s=timeout_s,
                    memory_bytes=memory_bytes,
                    repetitions=repetitions,
                )
                row = {
                    **metadata,
                    **measured,
                    "matches_consensus": None,
                    "repetitions": repetitions,
                    "timeout_s": timeout_s,
                    "memory_bytes": memory_bytes,
                    "run_id": run_id,
                }
            if row["status"] in RESOURCE_FAILURES:
                blocked.add(workload_series)
        rows.append(row)
        if checkpoint_path is not None:
            write_csv(rows, checkpoint_path)
    return rows


def mark_against_saved_consensus(
    boundary_rows: list[dict[str, object]], saved_rows: Iterable[Mapping[str, object]]
) -> None:
    successful: dict[tuple[str, int], set[str]] = {}
    for row in saved_rows:
        if row["status"] == "ok":
            successful.setdefault(
                (str(row["source_sha256"]), int(row["domain_size"])), set()
            ).add(str(row["result"]))
    for row in boundary_rows:
        if row["status"] != "ok":
            row["matches_consensus"] = None
            continue
        expected = successful.get(
            (str(row["source_sha256"]), int(row["domain_size"]))
        )
        row["matches_consensus"] = (
            str(row["result"]) in expected
            if expected is not None and len(expected) == 1
            else None
        )


def _configs(rows: Iterable[Mapping[str, object]]) -> list[tuple[str, str]]:
    preferred = [
        (BRANCH, ALGORITHM),
        ("devel", "fastv2"),
        ("devel", "incremental3"),
        ("modk", "fastv2"),
        ("modk", "incremental3"),
    ]
    present = {(str(row["branch"]), str(row["algorithm"])) for row in rows}
    return [config for config in preferred if config in present]


def paired_stats(
    rows: Iterable[Mapping[str, object]], comparison: tuple[str, str]
) -> dict[str, float | int] | None:
    grouped: dict[tuple[str, int], dict[tuple[str, str], Mapping[str, object]]] = {}
    for row in rows:
        if row["status"] == "ok":
            grouped.setdefault(
                (str(row["source_sha256"]), int(row["domain_size"])), {}
            )[(str(row["branch"]), str(row["algorithm"]))] = row
    ratios: list[float] = []
    for group in grouped.values():
        if (BRANCH, ALGORITHM) not in group or comparison not in group:
            continue
        boundary_time = float(group[(BRANCH, ALGORITHM)]["solver_time_s"])
        old_time = float(group[comparison]["solver_time_s"])
        if boundary_time > 0 and old_time > 0:
            ratios.append(old_time / boundary_time)
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
    source_commit: str,
    timeout_s: float,
    memory_bytes: int,
) -> None:
    configs = _configs(rows)
    boundary_commit = next(
        str(row["commit"])
        for row in rows
        if row["branch"] == BRANCH and row["algorithm"] == ALGORITHM
    )
    lines = [
        "# Historical boundary-profile performance comparison",
        "",
        f"- Boundary-profile commit: `{boundary_commit}`",
        f"- Saved workload inventory commit: `{source_commit}`",
        f"- Limits: {timeout_s:g} seconds and {memory_bytes / 1024**3:g} GiB RSS per measurement",
        "- The four historical configurations were reused from `../cross_branch/results.csv`; they were not rerun.",
        "- Do not use this report as a current head-to-head comparison; use `domain_series_performance.py`.",
        "",
        "## Status totals",
        "",
        "| configuration | ok | timeout | memory | unsupported | invalid | error | skipped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in configs:
        selected = [
            row for row in rows
            if (str(row["branch"]), str(row["algorithm"])) == config
        ]
        count = lambda status: sum(row["status"] == status for row in selected)
        errors = sum(
            row["status"] not in (
                "ok", "timeout", "memory", "unsupported", "invalid",
                "skipped-after-resource",
            )
            for row in selected
        )
        lines.append(
            f"| {config[0]}/{config[1]} | {count('ok')} | {count('timeout')} | "
            f"{count('memory')} | {count('unsupported')} | {count('invalid')} | "
            f"{errors} | {count('skipped-after-resource')} |"
        )

    boundary = [
        row for row in rows
        if row["branch"] == BRANCH and row["algorithm"] == ALGORITHM
    ]
    compared = [row for row in boundary if row.get("matches_consensus") is not None]
    mismatches = [row for row in compared if row.get("matches_consensus") is False]
    lines.extend(["", "## Correctness", ""])
    lines.append(
        f"Boundary-profile matched the saved successful-result consensus on "
        f"{len(compared) - len(mismatches)}/{len(compared)} comparable workloads."
    )
    if mismatches:
        lines.append(f"Mismatches: {len(mismatches)}; inspect `results.csv`.")

    lines.extend([
        "", "## Paired solve-time comparison", "",
        "Ratios are `historical solve time / boundary-profile solve time`; values above 1 favor boundary-profile.",
        "",
        "| historical configuration | pairs | geometric mean | median | boundary-profile faster |",
        "|---|---:|---:|---:|---:|",
    ])
    for config in configs[1:]:
        stats = paired_stats(rows, config)
        if stats:
            lines.append(
                f"| {config[0]}/{config[1]} | {stats['pairs']} | "
                f"{stats['geomean']:.3f}x | {stats['median']:.3f}x | "
                f"{stats['boundary_faster_pct']:.1f}% |"
            )

    lines.extend([
        "", "## Successful-run peak RSS", "",
        "| configuration | median | maximum |", "|---|---:|---:|",
    ])
    for config in configs:
        values = [
            float(row["peak_rss_mib"])
            for row in rows
            if (str(row["branch"]), str(row["algorithm"])) == config
            and row["status"] == "ok"
        ]
        if values:
            lines.append(
                f"| {config[0]}/{config[1]} | {statistics.median(values):.1f} MiB | "
                f"{max(values):.1f} MiB |"
            )
    path.write_text("\n".join(lines) + "\n")


def plot_metric(
    rows: list[dict[str, object]], path: Path, metric: str, ylabel: str
) -> None:
    import matplotlib.pyplot as plt

    successful = [
        row for row in rows
        if row["status"] == "ok" and row.get(metric) not in (None, "")
    ]
    series = sorted({(str(row["family"]), str(row["variant"])) for row in successful})
    if not series:
        return
    columns = 3
    fig, axes = plt.subplots(
        math.ceil(len(series) / columns), columns,
        figsize=(18, max(5, 4 * math.ceil(len(series) / columns))), squeeze=False,
    )
    styles = {
        (BRANCH, ALGORITHM): ("#2ca02c", "^", "-"),
        ("devel", "fastv2"): ("#1f77b4", "o", "-"),
        ("devel", "incremental3"): ("#ff7f0e", "s", "-"),
        ("modk", "fastv2"): ("#1f77b4", "o", "--"),
        ("modk", "incremental3"): ("#ff7f0e", "s", "--"),
    }
    for axis, name in zip(axes.flat, series):
        group = [row for row in successful if (row["family"], row["variant"]) == name]
        for config, (color, marker, linestyle) in styles.items():
            points = sorted(
                (int(row["domain_size"]), float(row[metric]))
                for row in group
                if (row["branch"], row["algorithm"]) == config
            )
            if points:
                axis.plot(
                    [point[0] for point in points], [point[1] for point in points],
                    color=color, marker=marker, linestyle=linestyle,
                    label=f"{config[0]}/{config[1]}",
                )
        axis.set_title(name[0] + (f" [{name[1]}]" if name[1] != "default" else ""), fontsize=9)
        axis.set_xlabel("domain size")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    for axis in axes.flat[len(series):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saved-results", type=Path,
        default=ROOT / "benchmarks/results/cross_branch/results.csv",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "benchmarks/results/boundary_profile",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--memory-gib", type=float, default=4.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--limit-workloads", type=int, default=None, help="smoke/debug only")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0 or args.memory_gib <= 0 or args.repetitions < 1:
        raise SystemExit("timeout, memory, and repetitions must be positive")
    saved_rows = read_csv(args.saved_results)
    workloads, source_commit = reconstruct_workloads(saved_rows)
    if args.limit_workloads is not None:
        workloads = workloads[: args.limit_workloads]
    commit = resolve_commit("HEAD")
    args.out.mkdir(parents=True, exist_ok=True)
    input_dir = args.out / ".inputs"
    input_dir.mkdir(exist_ok=True)
    result_path = args.out / "results.csv"
    resume = [] if args.no_resume else read_csv(result_path)
    memory_bytes = int(args.memory_gib * 1024**3)
    environments = {BRANCH: (Path(sys.executable), commit)}
    run_id = benchmark_run_id(
        workloads, environments, algorithms=(ALGORITHM,),
        timeout_s=args.timeout, memory_bytes=memory_bytes,
        repetitions=args.repetitions,
    )
    boundary_rows = execute_boundary_profile(
        workloads,
        python=Path(sys.executable),
        commit=commit,
        timeout_s=args.timeout,
        memory_bytes=memory_bytes,
        repetitions=args.repetitions,
        input_dir=input_dir,
        checkpoint_path=result_path,
        resume_rows=resume,
        run_id=run_id,
    )
    mark_against_saved_consensus(boundary_rows, saved_rows)
    write_csv(boundary_rows, result_path)
    combined = [dict(row) for row in saved_rows] + boundary_rows
    write_csv(combined, args.out / "combined_results.csv")
    write_summary(
        combined, args.out / "summary.md", source_commit=source_commit,
        timeout_s=args.timeout, memory_bytes=memory_bytes,
    )
    plot_metric(combined, args.out / "runtime.png", "solver_time_s", "solve time (s)")
    plot_metric(combined, args.out / "memory.png", "peak_rss_mib", "peak RSS (MiB)")
    print(f"Wrote {len(boundary_rows)} boundary-profile rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
