#!/usr/bin/env python3
"""Compare devel/modk fastv2 and incremental3 with hard resource bounds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import psutil

from benchmarks.cases import BenchmarkCase, benchmark_cases


ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("cross_branch_worker.py")
SENTINEL = "BENCH_RESULT_JSON="
RESOURCE_FAILURES = frozenset(("timeout", "memory"))
ALGORITHMS = ("fastv2", "incremental3")
MODEL_DOMAIN_SIZES = (2, 4, 8, 16, 32, 64)
RESULT_FIELDS = (
    "source_kind", "case", "family", "category", "variant", "domain_size",
    "source_sha256", "branch", "commit", "algorithm", "status",
    "solver_time_s", "parse_time_s", "wall_time_s", "peak_rss_bytes", "peak_rss_mib",
    "result", "matches_consensus", "repetitions", "error", "stderr",
)
_INTEGER_DOMAIN_RE = re.compile(
    r"^(?P<prefix>\s*[A-Za-z][A-Za-z0-9_]*\s*=\s*)(?P<size>\d+)"
    r"(?P<suffix>\s*(?:#.*)?)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Workload:
    source_kind: str
    case: str
    family: str
    category: str
    variant: str
    domain_size: int
    suffix: str
    source: str

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()


def rewrite_integer_domain(source: str, domain_size: int) -> str:
    """Replace the single integer domain declaration in WFOMCS/MLN text."""

    matches = list(_INTEGER_DOMAIN_RE.finditer(source))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one integer domain declaration, found {len(matches)}")
    match = matches[0]
    return source[: match.start()] + match["prefix"] + str(domain_size) + match["suffix"] + source[match.end() :]


def _number(value: object) -> str:
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator not in (None, 1):
        return f"{numerator}/{denominator}"
    return str(value)


def _serialize_formula(formula: object) -> str:
    """Serialize the current typed AST using syntax shared with legacy modk."""

    from wfomc.fol import (
        And,
        Atom,
        BoolConst,
        CountingQuantifier,
        Eq,
        Iff,
        Implies,
        Not,
        Or,
        Quantifier,
        QuantifierKind,
    )

    if isinstance(formula, Atom):
        terms = ",".join(str(term) for term in formula.terms)
        return f"{formula.predicate}({terms})" if formula.terms else str(formula.predicate)
    if isinstance(formula, BoolConst):
        raise ValueError("the shared legacy DSL cannot serialize Boolean constants")
    if isinstance(formula, Eq):
        return f"{formula.left} = {formula.right}"
    if isinstance(formula, Not):
        return f"~({_serialize_formula(formula.body)})"
    if isinstance(formula, (And, Or)):
        operator = " & " if isinstance(formula, And) else " | "
        return operator.join(f"({_serialize_formula(item)})" for item in formula.args)
    if isinstance(formula, Implies):
        return f"({_serialize_formula(formula.left)}) -> ({_serialize_formula(formula.right)})"
    if isinstance(formula, Iff):
        return f"({_serialize_formula(formula.left)}) <-> ({_serialize_formula(formula.right)})"
    if isinstance(formula, Quantifier):
        keyword = "\\forall" if formula.kind == QuantifierKind.FORALL else "\\exists"
        body = _serialize_formula(formula.body)
        for variable in reversed(formula.variables):
            body = f"{keyword} {variable}: ({body})"
        return body
    if isinstance(formula, CountingQuantifier):
        comparator = formula.comparator
        count = formula.count
        if comparator == "mod":
            if isinstance(count, tuple):
                spec = f"{count[0]}mod{count[1]}"
            else:
                spec = f"{count.remainder}mod{count.modulus}"
        else:
            spec = f"{comparator}{count}"
        return f"\\exists_{{{spec}}} {formula.variable}: ({_serialize_formula(formula.body)})"
    raise TypeError(f"unsupported formula node: {type(formula).__name__}")


def serialize_catalog_case(case: BenchmarkCase) -> str:
    """Render a typed catalog case into the legacy-compatible WFOMCS DSL."""

    problem = case.build_problem()
    lines = [_serialize_formula(problem.sentence), f"D = {case.domain_size}"]
    for predicate, (positive, negative) in sorted(
        problem.weights.items(), key=lambda item: str(item[0])
    ):
        lines.append(f"{_number(positive)} {_number(negative)} {predicate}")
    for constraint in problem.cardinality_constraints.constraints:
        pieces: list[str] = []
        for index, term in enumerate(constraint.terms):
            coefficient = term.coefficient
            atom = f"|{term.predicate}|" if coefficient == 1 else f"{_number(coefficient)} |{term.predicate}|"
            if index and not str(coefficient).startswith("-"):
                atom = "+ " + atom
            pieces.append(atom)
        lines.append(
            f"{' '.join(pieces)} {constraint.comparator.value} {_number(constraint.rhs)}"
        )
    if problem.evidence.unary.literals or problem.evidence.binary.literals:
        raise ValueError(f"catalog case {case.key} unexpectedly contains evidence")
    return "\n".join(lines) + "\n"


def parse_worker_output(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(SENTINEL):
            value = json.loads(line[len(SENTINEL) :])
            if not isinstance(value, dict):
                raise ValueError("worker payload is not an object")
            return value
    raise ValueError("worker output did not contain the result sentinel")


def truncation_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["source_kind"], row["family"], row["variant"],
        row["branch"], row["algorithm"],
    )


def series_key(row: Mapping[str, object]) -> tuple[object, ...]:
    """Identify one increasing-domain problem series across all configurations."""

    return row["source_kind"], row["family"], row["variant"]


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(("git", *args), cwd=cwd, text=True).strip()


def resolve_commit(ref: str) -> str:
    return _git("rev-parse", ref)


def prepare_modk_worktree(ref: str, *, sync: bool = True) -> tuple[Path, str]:
    commit = resolve_commit(ref)
    path = Path(tempfile.gettempdir()) / f"wfomc-cross-branch-{commit[:12]}"
    if path.exists():
        try:
            if _git("rev-parse", "HEAD", cwd=path) != commit:
                raise RuntimeError(f"existing worktree {path} is at the wrong commit")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(f"existing path is not the expected worktree: {path}") from None
    else:
        subprocess.run(
            ("git", "worktree", "add", "--detach", str(path), commit),
            cwd=ROOT, check=True,
        )
    python = path / ".venv" / "bin" / "python"
    if sync and not python.exists():
        subprocess.run(("uv", "sync", "--project", str(path)), check=True)
    if sync and not python.exists():
        raise RuntimeError(f"uv did not create {python}")
    return path, commit


def collect_catalog_workloads(suite: str) -> list[Workload]:
    rows: list[Workload] = []
    for case in benchmark_cases(suite):
        rows.append(
            Workload(
                source_kind="catalog",
                case=case.key,
                family=f"{case.category}/{case.family}",
                category=case.category,
                variant=case.variant,
                domain_size=case.domain_size,
                suffix=".wfomcs",
                source=serialize_catalog_case(case),
            )
        )
    return rows


def _tree_models(commit: str) -> set[str]:
    output = _git("ls-tree", "-r", "--name-only", commit, "--", "models")
    return {
        line for line in output.splitlines()
        if line.endswith((".wfomcs", ".mln"))
    }


def collect_model_workloads(
    current_commit: str,
    baseline_commit: str,
) -> tuple[list[Workload], list[dict[str, str]]]:
    common = sorted(_tree_models(current_commit) & _tree_models(baseline_commit))
    workloads: list[Workload] = []
    exclusions: list[dict[str, str]] = []
    for relative in common:
        path = ROOT / relative
        source = path.read_text()
        if len(list(_INTEGER_DOMAIN_RE.finditer(source))) != 1:
            exclusions.append({"model": relative, "reason": "domain is fixed/named or ambiguous"})
            continue
        # Keep the extension: e.g. friends-smokes.mln and
        # friends-smokes.wfomcs are distinct problem series.
        family = relative
        for domain_size in MODEL_DOMAIN_SIZES:
            workloads.append(
                Workload(
                    source_kind="model",
                    case=f"{relative}/n{domain_size}",
                    family=family,
                    category="model",
                    variant="default",
                    domain_size=domain_size,
                    suffix=path.suffix,
                    source=rewrite_integer_domain(source, domain_size),
                )
            )
    current_only = sorted(_tree_models(current_commit) - _tree_models(baseline_commit))
    baseline_only = sorted(_tree_models(baseline_commit) - _tree_models(current_commit))
    exclusions.extend({"model": path, "reason": "current-branch only"} for path in current_only)
    exclusions.extend({"model": path, "reason": "baseline-branch only"} for path in baseline_only)
    return workloads, exclusions


def _rss_for_tree(process: psutil.Process) -> int:
    total = 0
    for item in (process, *process.children(recursive=True)):
        try:
            total += item.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def run_worker(
    python: Path,
    workload_path: Path,
    algorithm: str,
    *,
    timeout_s: float,
    memory_bytes: int,
    repetitions: int,
) -> dict[str, object]:
    started = time.perf_counter()
    process = subprocess.Popen(
        (
            str(python), str(WORKER), "--input", str(workload_path),
            "--algorithm", algorithm, "--repetitions", str(repetitions),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    ps_process = psutil.Process(process.pid)
    peak_rss = 0
    forced_status: str | None = None
    limit = timeout_s * repetitions
    try:
        while process.poll() is None:
            try:
                peak_rss = max(peak_rss, _rss_for_tree(ps_process))
            except psutil.NoSuchProcess:
                pass
            elapsed = time.perf_counter() - started
            if peak_rss > memory_bytes:
                forced_status = "memory"
                _kill_process_group(process)
                break
            if elapsed > limit:
                forced_status = "timeout"
                _kill_process_group(process)
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        _kill_process_group(process)
        process.communicate()
        raise
    stdout_bytes, stderr_bytes = process.communicate()
    wall_time = time.perf_counter() - started
    stdout = stdout_bytes.decode(errors="replace")
    stderr = stderr_bytes.decode(errors="replace")
    try:
        peak_rss = max(peak_rss, _rss_for_tree(ps_process))
    except psutil.NoSuchProcess:
        pass
    base = {
        "wall_time_s": wall_time,
        "peak_rss_bytes": peak_rss,
        "peak_rss_mib": peak_rss / (1024 * 1024),
        "stderr": stderr[-4000:],
    }
    if forced_status:
        return {
            **base,
            "status": forced_status,
            "result": None,
            "solver_time_s": None,
            "error": (
                f"exceeded {memory_bytes / (1024**3):.1f} GiB RSS"
                if forced_status == "memory"
                else f"exceeded {limit:.1f}s wall time"
            ),
        }
    try:
        payload = parse_worker_output(stdout)
    except (ValueError, json.JSONDecodeError) as error:
        return {
            **base, "status": "worker-error", "result": None,
            "solver_time_s": None, "error": f"{type(error).__name__}: {error}",
        }
    return {**base, **payload}


def _write_workload(workload: Workload, directory: Path) -> Path:
    digest = workload.source_sha256[:16]
    path = directory / f"{digest}{workload.suffix}"
    if not path.exists():
        path.write_text(workload.source)
    return path


def _skipped_row(
    workload: Workload,
    branch: str,
    commit: str,
    algorithm: str,
    repetitions: int,
) -> dict[str, object]:
    return {
        **asdict(workload), "source": None, "source_sha256": workload.source_sha256,
        "branch": branch, "commit": commit, "algorithm": algorithm,
        "status": "skipped-after-resource", "solver_time_s": None,
        "parse_time_s": None,
        "wall_time_s": None, "peak_rss_bytes": None, "peak_rss_mib": None,
        "result": None, "matches_consensus": None, "repetitions": repetitions,
        "error": "a smaller domain in this series hit a resource limit", "stderr": "",
    }


def execute(
    workloads: Iterable[Workload],
    environments: Mapping[str, tuple[Path, str]],
    *,
    timeout_s: float,
    memory_bytes: int,
    repetitions: int,
    input_dir: Path,
    checkpoint_path: Path | None = None,
    resume_rows: Iterable[dict[str, object]] = (),
) -> list[dict[str, object]]:
    ordered = sorted(
        workloads,
        key=lambda item: (item.source_kind, item.family, item.variant, item.domain_size, item.case),
    )
    rows: list[dict[str, object]] = []
    blocked: set[tuple[object, ...]] = set()
    resume_by_key = {
        (
            str(row["source_sha256"]), int(row["domain_size"]),
            str(row["branch"]), str(row["algorithm"]),
        ): row
        for row in resume_rows
        if row.get("status") != "skipped-after-resource"
    }
    total = len(ordered) * len(environments) * len(ALGORITHMS)
    index = 0
    for workload in ordered:
        path = _write_workload(workload, input_dir)
        workload_key = series_key(asdict(workload))
        resource_failed = False
        for branch, (python, commit) in environments.items():
            for algorithm in ALGORITHMS:
                index += 1
                metadata = {
                    "source_kind": workload.source_kind, "case": workload.case,
                    "family": workload.family, "category": workload.category,
                    "variant": workload.variant, "domain_size": workload.domain_size,
                    "source_sha256": workload.source_sha256, "branch": branch,
                    "commit": commit, "algorithm": algorithm,
                }
                if workload_key in blocked:
                    row = _skipped_row(workload, branch, commit, algorithm, repetitions)
                else:
                    resume_key = (
                        workload.source_sha256, workload.domain_size, branch, algorithm,
                    )
                    previous = resume_by_key.get(resume_key)
                    if previous is not None:
                        row = {
                            **previous, **metadata,
                            "matches_consensus": None,
                        }
                    else:
                        print(
                            f"[{index}/{total}] {branch}/{algorithm} {workload.case}",
                            flush=True,
                        )
                        measured = run_worker(
                            python, path, algorithm, timeout_s=timeout_s,
                            memory_bytes=memory_bytes, repetitions=repetitions,
                        )
                        row = {
                            **metadata, **measured, "matches_consensus": None,
                            "repetitions": repetitions,
                        }
                    if row["status"] in RESOURCE_FAILURES:
                        resource_failed = True
                rows.append(row)
        if resource_failed:
            blocked.add(workload_key)
        if checkpoint_path is not None:
            mark_consensus(rows)
            write_csv(rows, checkpoint_path)
    mark_consensus(rows)
    return rows


def mark_consensus(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["source_sha256"], row["domain_size"]), []
        ).append(row)
    for group in grouped.values():
        successful = [row for row in group if row["status"] == "ok"]
        results = {str(row["result"]) for row in successful}
        for row in successful:
            row["matches_consensus"] = len(results) == 1


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_exclusions(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "reason"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _format_time(value: object) -> str:
    if value in (None, ""):
        return "-"
    seconds = float(value)
    return f"{seconds * 1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


def paired_branch_stats(
    rows: list[dict[str, object]], algorithm: str
) -> dict[str, float | int] | None:
    """Summarize modk/devel solve-time ratios for successful paired rows."""

    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["source_sha256"]), int(row["domain_size"])), []
        ).append(row)
    ratios: list[float] = []
    for group in grouped.values():
        selected = {
            str(row["branch"]): row
            for row in group
            if row["algorithm"] == algorithm and row["status"] == "ok"
        }
        if set(selected) != {"devel", "modk"}:
            continue
        devel_time = float(selected["devel"]["solver_time_s"])
        modk_time = float(selected["modk"]["solver_time_s"])
        if devel_time > 0 and modk_time > 0:
            ratios.append(modk_time / devel_time)
    if not ratios:
        return None
    return {
        "pairs": len(ratios),
        "geomean": math.exp(statistics.fmean(math.log(ratio) for ratio in ratios)),
        "median": statistics.median(ratios),
        "devel_faster_pct": 100 * sum(ratio > 1 for ratio in ratios) / len(ratios),
    }


def write_summary(
    rows: list[dict[str, object]],
    exclusions: list[dict[str, str]],
    path: Path,
    *,
    timeout_s: float,
    memory_bytes: int,
) -> None:
    configs = [(branch, algo) for branch in ("devel", "modk") for algo in ALGORITHMS]
    commits = {str(row["branch"]): str(row["commit"]) for row in rows}
    lines = [
        "# Cross-branch WFOMC performance comparison", "",
        f"- `devel`: `{commits.get('devel', '-')}`",
        f"- `modk`: `{commits.get('modk', '-')}`",
        f"- Limits: {timeout_s:g} seconds and {memory_bytes / (1024**3):g} GiB RSS per measurement",
        f"- Rows: {len(rows)}; excluded model files: {len(exclusions)}", "",
        "## Status totals", "",
        "| configuration | ok | timeout | memory | error | skipped |", "|---|---:|---:|---:|---:|---:|",
    ]
    for branch, algorithm in configs:
        selected = [row for row in rows if row["branch"] == branch and row["algorithm"] == algorithm]
        counts = {status: sum(row["status"] == status for row in selected) for status in ("ok", "timeout", "memory")}
        errors = sum(row["status"] not in ("ok", "timeout", "memory", "skipped-after-resource") for row in selected)
        skipped = sum(row["status"] == "skipped-after-resource" for row in selected)
        lines.append(f"| {branch}/{algorithm} | {counts['ok']} | {counts['timeout']} | {counts['memory']} | {errors} | {skipped} |")

    mismatches = [row for row in rows if row.get("matches_consensus") is False]
    lines.extend(["", "## Correctness", ""])
    if mismatches:
        groups = {(row["case"], row["domain_size"]) for row in mismatches}
        lines.append(f"⚠ {len(groups)} workload/domain groups have differing successful results; inspect `results.csv`.")
    else:
        lines.append("All groups with two or more successful configurations returned the same result.")

    lines.extend([
        "", "## Aggregate paired branch comparison", "",
        "Ratios are `modk solve time / devel solve time`; values above 1 favor `devel`.", "",
        "| algorithm | paired workloads | geometric mean | median | devel faster |",
        "|---|---:|---:|---:|---:|",
    ])
    for algorithm in ALGORITHMS:
        stats = paired_branch_stats(rows, algorithm)
        if stats is not None:
            lines.append(
                f"| {algorithm} | {stats['pairs']} | {stats['geomean']:.3f}x | "
                f"{stats['median']:.3f}x | {stats['devel_faster_pct']:.1f}% |"
            )

    lines.extend([
        "", "## Successful-run peak RSS", "",
        "| configuration | median | maximum |", "|---|---:|---:|",
    ])
    for branch, algorithm in configs:
        memory_values = [
            float(row["peak_rss_mib"])
            for row in rows
            if row["branch"] == branch
            and row["algorithm"] == algorithm
            and row["status"] == "ok"
        ]
        if memory_values:
            lines.append(
                f"| {branch}/{algorithm} | {statistics.median(memory_values):.1f} MiB | "
                f"{max(memory_values):.1f} MiB |"
            )

    lines.extend([
        "", "## Largest commonly successful domain per series", "",
        "| series | n | devel/fastv2 | devel/incremental3 | modk/fastv2 | modk/incremental3 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    by_series: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        by_series.setdefault((str(row["family"]), str(row["variant"])), []).append(row)
    for series, group in sorted(by_series.items()):
        domains = sorted({int(row["domain_size"]) for row in group}, reverse=True)
        selected_domain = None
        cells: list[str] = []
        for domain in domains:
            candidate = []
            for branch, algorithm in configs:
                match = next((row for row in group if row["domain_size"] == domain and row["branch"] == branch and row["algorithm"] == algorithm), None)
                if match is None or match["status"] != "ok":
                    break
                candidate.append(_format_time(match["solver_time_s"]))
            if len(candidate) == len(configs):
                selected_domain, cells = domain, candidate
                break
        if selected_domain is not None:
            label = series[0] + (f" [{series[1]}]" if series[1] != "default" else "")
            lines.append(f"| {label} | {selected_domain} | {' | '.join(cells)} |")
    path.write_text("\n".join(lines) + "\n")


def plot_metric(rows: list[dict[str, object]], path: Path, metric: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    successful = [row for row in rows if row["status"] == "ok" and row.get(metric) not in (None, "")]
    series_names = sorted({(str(row["family"]), str(row["variant"])) for row in successful})
    if not series_names:
        return
    columns = 3
    rows_count = math.ceil(len(series_names) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(18, max(5, 4 * rows_count)), squeeze=False)
    styles = {
        ("devel", "fastv2"): ("#1f77b4", "o", "-"),
        ("devel", "incremental3"): ("#ff7f0e", "s", "-"),
        ("modk", "fastv2"): ("#1f77b4", "o", "--"),
        ("modk", "incremental3"): ("#ff7f0e", "s", "--"),
    }
    for axis, series in zip(axes.flat, series_names):
        family, variant = series
        group = [row for row in successful if row["family"] == family and row["variant"] == variant]
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
        axis.set_title(family + (f" [{variant}]" if variant != "default" else ""), fontsize=9)
        axis.set_xlabel("domain size")
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    for axis in axes.flat[len(series_names) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-ref", default="HEAD")
    parser.add_argument("--baseline-ref", default="modk")
    parser.add_argument("--suite", default="all")
    parser.add_argument("--sources", choices=("all", "catalog", "models"), default="all")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--memory-gib", type=float, default=4.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--limit-workloads", type=int, default=None, help="smoke/debug only")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks/results/cross_branch")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0 or args.memory_gib <= 0 or args.repetitions < 1:
        raise SystemExit("timeout, memory, and repetitions must be positive")
    current_commit = resolve_commit(args.current_ref)
    modk_root, baseline_commit = prepare_modk_worktree(args.baseline_ref, sync=not args.skip_sync)
    current_python = Path(sys.executable)
    baseline_python = modk_root / ".venv/bin/python"
    if not baseline_python.exists():
        raise SystemExit(f"missing baseline Python: {baseline_python}")

    workloads: list[Workload] = []
    exclusions: list[dict[str, str]] = []
    if args.sources in ("all", "catalog"):
        workloads.extend(collect_catalog_workloads(args.suite))
    if args.sources in ("all", "models"):
        model_workloads, exclusions = collect_model_workloads(current_commit, baseline_commit)
        workloads.extend(model_workloads)
    if args.limit_workloads is not None:
        workloads = workloads[: args.limit_workloads]

    args.out.mkdir(parents=True, exist_ok=True)
    input_dir = args.out / ".inputs"
    input_dir.mkdir(exist_ok=True)
    write_exclusions(exclusions, args.out / "excluded.csv")
    checkpoint_path = args.out / "results.csv"
    resume_rows = [] if args.no_resume else read_csv(checkpoint_path)
    rows = execute(
        workloads,
        {
            "devel": (current_python, current_commit),
            "modk": (baseline_python, baseline_commit),
        },
        timeout_s=args.timeout,
        memory_bytes=int(args.memory_gib * 1024**3),
        repetitions=args.repetitions,
        input_dir=input_dir,
        checkpoint_path=checkpoint_path,
        resume_rows=resume_rows,
    )
    write_csv(rows, args.out / "results.csv")
    write_summary(
        rows, exclusions, args.out / "summary.md", timeout_s=args.timeout,
        memory_bytes=int(args.memory_gib * 1024**3),
    )
    plot_metric(rows, args.out / "runtime.png", "solver_time_s", "solve time (s)")
    plot_metric(rows, args.out / "memory.png", "peak_rss_mib", "peak RSS (MiB)")
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
