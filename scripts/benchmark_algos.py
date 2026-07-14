#!/usr/bin/env python3
"""Benchmark every WFOMC algorithm across all in-scope models.

Iterates ``models/``, ``models/unary_evidence/``, ``models/linear_order/``,
``models/linear_order/unary_evidence/``, and ``models/linear_order/predk/``.
Models that use ``CIRCULAR_PRED`` are skipped because the propositional counter
does not yet support them. For every remaining model, this script runs each
algorithm and records the median wall-clock time. Results are written to::

    check-points/benchmark.csv          per-(model, algo) timings + status
    check-points/benchmark_summary.md   markdown summary table
    check-points/benchmark.png          grouped bar chart (log y)

Usage::

    GANAK=/path/to/ganak uv run python scripts/benchmark_algos.py [--trials N] [--timeout S]

If ganak is not on PATH and ``GANAK`` is unset, the propositional rows are
marked as skipped.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

logger.disable("wfomc")

from wfomc import Algo, parse_input, wfomc                       # noqa: E402
from wfomc.ganak import GanakError, find_ganak                 # noqa: E402


ALGOS: list[Algo] = [
    Algo.STANDARD,
    Algo.FAST,
    Algo.FASTv2,
    Algo.INCREMENTAL,
    Algo.RECURSIVE,
    Algo.PROPOSITIONAL,
]

ROOT: Path = Path(__file__).resolve().parent.parent
MODEL_DIRS: list[Path] = [
    ROOT / "models",
    ROOT / "models" / "unary_evidence",
    ROOT / "models" / "linear_order",
    ROOT / "models" / "linear_order" / "unary_evidence",
    ROOT / "models" / "linear_order" / "predk",
]


class _Timeout(Exception):
    pass


@contextmanager
def time_budget(seconds: float):
    """Hard wall-clock timeout via SIGALRM."""
    def _handler(signum, frame):
        raise _Timeout(f"exceeded {seconds:.1f}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 0.001))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


def collect_models() -> list[Path]:
    files: list[Path] = []
    for d in MODEL_DIRS:
        if not d.exists():
            continue
        files.extend(sorted(d.glob("*.wfomcs")))
        files.extend(sorted(d.glob("*.mln")))
    return files


def run_one(problem, algo: Algo, trials: int, timeout: float) -> dict:
    """Run a single (problem, algo) pair, returning a result dict."""
    times: list[float] = []
    result = None
    status = "ok"
    err: str | None = None
    for _ in range(trials):
        t0 = time.perf_counter()
        try:
            with time_budget(timeout):
                res = wfomc(problem, algo=algo)
        except _Timeout as e:
            status, err = "timeout", str(e)
            break
        except Exception as e:  # noqa: BLE001
            status, err = "error", f"{type(e).__name__}: {e}"
            break
        times.append(time.perf_counter() - t0)
        result = res
    median = statistics.median(times) if times else None
    return {"time": median, "result": result, "status": status, "error": err}


def benchmark(trials: int, timeout: float, ganak_available: bool) -> list[dict]:
    rows: list[dict] = []
    for model in collect_models():
        # Parse once to inspect the problem; reparse per call so each algo
        # gets a fresh problem (some algos mutate via deepcopy in context).
        try:
            probe = parse_input(str(model))
        except Exception as e:  # noqa: BLE001
            print(f"[parse-error] {model.name}: {e}", file=sys.stderr)
            continue
        if probe.contain_circular_predecessor_axiom():
            continue  # CIRCULAR_PRED not supported by the propositional counter

        domain_size = len(probe.domain)
        print(f"\n=== {model.relative_to(ROOT)}  |D|={domain_size} ===", flush=True)
        ref_value = None

        for algo in ALGOS:
            row = {
                "model": str(model.relative_to(ROOT)),
                "algo": algo.value,
                "domain_size": domain_size,
                "time": None,
                "result": None,
                "matches_ref": None,
                "status": "skipped",
                "error": None,
            }
            if algo is Algo.PROPOSITIONAL and not ganak_available:
                row["error"] = "ganak not available"
                rows.append(row)
                print(f"  {algo.value:<14}  SKIP (ganak not available)", flush=True)
                continue

            problem = parse_input(str(model))
            res = run_one(problem, algo, trials, timeout)
            row["time"] = res["time"]
            row["status"] = res["status"]
            row["error"] = res["error"]
            if res["status"] == "ok":
                row["result"] = str(res["result"])
                # Pick FASTv2 if it succeeded; otherwise fall back to
                # INCREMENTAL (which handles LEQ / PREk).
                if ref_value is None and algo in (Algo.FASTv2, Algo.INCREMENTAL):
                    ref_value = res["result"]
                row["matches_ref"] = (
                    None if ref_value is None else res["result"] == ref_value
                )
            tstr = f"{res['time']*1000:8.2f} ms" if res["time"] is not None else "    -    "
            print(f"  {algo.value:<14}  {tstr}  {res['status']}"
                  + (f"  ({res['error']})" if res["error"] else ""),
                  flush=True)
            rows.append(row)

        # Fill matches_ref retroactively if FASTv2 came later than another algo
        # (it always runs in our ALGOS order before propositional but after the
        # others; recompute against the now-known ref).
        if ref_value is not None:
            for row in rows:
                if row["model"] == str(model.relative_to(ROOT)) \
                        and row["status"] == "ok" \
                        and row["matches_ref"] is None:
                    # parse the recorded string back to compare structurally
                    try:
                        row["matches_ref"] = str(ref_value) == row["result"]
                    except Exception:  # noqa: BLE001
                        row["matches_ref"] = False
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    fields = ["model", "algo", "domain_size", "time", "status",
              "matches_ref", "result", "error"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def write_summary(rows: list[dict], out: Path) -> None:
    models = sorted({r["model"] for r in rows})
    algos = [a.value for a in ALGOS]
    lines = ["# WFOMC algorithm benchmark", ""]
    header = "| model | " + " | ".join(algos) + " |"
    sep = "|" + "|".join(["---"] * (len(algos) + 1)) + "|"
    lines += [header, sep]
    for model in models:
        cells: list[str] = [model]
        for algo in algos:
            cell = next((r for r in rows if r["model"] == model and r["algo"] == algo), None)
            if cell is None or cell["status"] != "ok":
                tag = (cell or {}).get("status", "-")
                cells.append(tag)
                continue
            t = cell["time"]
            tag = f"{t*1000:.1f} ms" if t < 1 else f"{t:.2f} s"
            if cell.get("matches_ref") is False:
                tag = f"⚠ {tag}"
            cells.append(tag)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("⚠ = result disagrees with `fastv2` reference (pre-existing bug suspected).")
    out.write_text("\n".join(lines) + "\n")


def plot_results(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    algos = [a.value for a in ALGOS]
    # Order models by sum-of-medians (smallest workloads first)
    by_model: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["status"] == "ok" and r["time"] is not None:
            by_model.setdefault(r["model"], {})[r["algo"]] = r["time"]
    models = sorted(by_model, key=lambda m: sum(by_model[m].values()))
    if not models:
        print("Nothing to plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(max(11, 0.75 * len(models)), 6))
    x = np.arange(len(models))
    width = 0.85 / len(algos)
    cmap = plt.get_cmap("tab10")
    for i, algo in enumerate(algos):
        heights = [by_model.get(m, {}).get(algo, np.nan) for m in models]
        ax.bar(x + (i - (len(algos) - 1) / 2) * width, heights, width,
                label=algo, color=cmap(i))
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("models/", "") for m in models],
                        rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.set_title("WFOMC algorithm timing per model (median of multiple trials)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(ncol=len(algos), loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=3,
                    help="number of trials per (model, algo) pair (default: 3)")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-trial wall-clock timeout in seconds (default: 60)")
    ap.add_argument("--ganak", help="path to ganak binary; overrides $GANAK")
    ap.add_argument("--out", default="check-points",
                    help="output directory (default: check-points)")
    args = ap.parse_args()

    if args.ganak:
        os.environ["GANAK"] = args.ganak

    try:
        find_ganak()
        ganak_available = True
    except GanakError as e:
        print(f"WARN: {e}", file=sys.stderr)
        ganak_available = False

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = benchmark(args.trials, args.timeout, ganak_available)
    write_csv(rows, out_dir / "benchmark.csv")
    write_summary(rows, out_dir / "benchmark_summary.md")
    plot_results(rows, out_dir / "benchmark.png")

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_total = len(rows)
    disagree = [r for r in rows if r.get("matches_ref") is False]
    print(f"\nWrote {out_dir}/benchmark.{{csv,png}} and benchmark_summary.md")
    print(f"Runs: {n_ok}/{n_total} ok; disagreements with fastv2: {len(disagree)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
