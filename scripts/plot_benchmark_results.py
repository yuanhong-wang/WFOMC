#!/usr/bin/env python3
"""Create paper figures and a summary from a benchmark CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ALGORITHMS = ("boundary-profile", "fast", "incremental3")
LABELS = {
    "boundary-profile": "BP-DP",
    "fast": "Fast",
    "incremental3": "Incremental3",
}
COLORS = {
    "boundary-profile": "#0072B2",
    "fast": "#D55E00",
    "incremental3": "#009E73",
}
MARKERS = {
    "boundary-profile": "o",
    "fast": "s",
    "incremental3": "^",
}
CATEGORIES = ("core", "c2", "cardinality", "unary-cardinality")
CATEGORY_LABELS = {
    "core": "Core",
    "c2": "$\\mathbf{C}^2$-related",
    "cardinality": "Cardinality",
    "unary-cardinality": "Unary\ncardinality",
}
SCALING_FAMILIES = (
    (
        "3-regular",
        "c2",
        "undirected-3-regular",
        "fo2-cardinality-reduction",
    ),
    ("4-coloured", "core", "properly-4-coloured-graph", "default"),
    ("Derangements", "core", "derangements", "fo2-cardinality-reduction"),
    (
        "3-matchings",
        "core",
        "3-edge-disjoint-perfect-matchings",
        "fo2-cardinality-reduction",
    ),
)


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _geomean(values: Sequence[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    return math.exp(statistics.fmean(math.log(value) for value in positive))


def validate_results(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_cases: int = 165,
    allow_partial: bool = False,
) -> None:
    cases = {row["case"] for row in rows}
    algorithms = {row["algorithm"] for row in rows}
    expected_rows = expected_cases * len(ALGORITHMS)
    if allow_partial and len(cases) > expected_cases:
        raise ValueError(f"expected at most {expected_cases} cases, found {len(cases)}")
    if not allow_partial and len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} cases, found {len(cases)}")
    if algorithms != set(ALGORITHMS):
        raise ValueError(f"unexpected algorithms: {sorted(algorithms)}")
    if allow_partial and len(rows) > expected_rows:
        raise ValueError(f"expected at most {expected_rows} rows, found {len(rows)}")
    if not allow_partial and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(rows)}")
    keys = [(row["case"], row["algorithm"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("case/algorithm rows are not unique")


def paired_speedups(
    rows: Iterable[Mapping[str, str]],
    competitor: str,
) -> list[tuple[str, float]]:
    by_case: dict[str, dict[str, Mapping[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["algorithm"]] = row
    points = []
    for selected in by_case.values():
        bp = selected.get("boundary-profile")
        other = selected.get(competitor)
        if not bp or not other or bp["status"] != "ok" or other["status"] != "ok":
            continue
        bp_time = float(bp["solver_time_s"])
        other_time = float(other["solver_time_s"])
        if bp_time > 0 and other_time > 0:
            points.append((bp["category"], other_time / bp_time))
    return points


def observed_pair_outcomes(
    rows: Iterable[Mapping[str, str]],
    competitor: str,
) -> Counter[str]:
    """Count solve outcomes where both BP-DP and competitor jobs were observed."""

    by_case: dict[str, dict[str, Mapping[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["algorithm"]] = row
    outcomes: Counter[str] = Counter()
    for selected in by_case.values():
        bp = selected.get("boundary-profile")
        other = selected.get(competitor)
        if bp is None or other is None:
            continue
        bp_solved = bp["status"] == "ok"
        other_solved = other["status"] == "ok"
        if bp_solved and other_solved:
            outcomes["both_solved"] += 1
        elif bp_solved:
            outcomes["bp_dp_only"] += 1
        elif other_solved:
            outcomes["competitor_only"] += 1
        else:
            outcomes["neither_solved"] += 1
    return outcomes


def select_scaling_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    category: str,
    family: str,
    variant: str,
) -> list[Mapping[str, str]]:
    """Select one unambiguous FastWFOMC-style benchmark family."""

    return sorted(
        (
            row
            for row in rows
            if row["category"] == category
            and row["family"] == family
            and row["variant"] == variant
        ),
        key=lambda row: (int(row["domain_size"]), row["algorithm"]),
    )


def summarize(rows: Sequence[Mapping[str, str]]) -> dict[str, object]:
    algorithms: dict[str, object] = {}
    for algorithm in ALGORITHMS:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        counts = Counter(row["status"] for row in selected)
        successful_times = [
            float(row["solver_time_s"])
            for row in selected
            if row["status"] == "ok"
        ]
        item: dict[str, object] = {
            "label": LABELS[algorithm],
            "status_counts": dict(sorted(counts.items())),
            "solved": counts["ok"],
            "median_successful_time_s": (
                statistics.median(successful_times) if successful_times else None
            ),
        }
        if algorithm != "boundary-profile":
            points = paired_speedups(rows, algorithm)
            values = [value for _category, value in points]
            item["paired_with_bp_dp"] = {
                "pairs": len(values),
                "geometric_mean_competitor_over_bp_dp": _geomean(values),
                "median_competitor_over_bp_dp": (
                    statistics.median(values) if values else None
                ),
                "bp_dp_faster_cases": sum(value > 1 for value in values),
                "competitor_faster_cases": sum(value < 1 for value in values),
                "ties": sum(value == 1 for value in values),
                "by_category": {
                    category: {
                        "pairs": len(category_values),
                        "geometric_mean_competitor_over_bp_dp": _geomean(
                            category_values
                        ),
                    }
                    for category in CATEGORIES
                    if (
                        category_values := [
                            value
                            for point_category, value in points
                            if point_category == category
                        ]
                    )
                },
            }
        algorithms[algorithm] = item
    mismatch_cases = sorted(
        {
            row["case"]
            for row in rows
            if row.get("comparison_status") == "mismatch"
        }
    )
    equivalence_mismatches = sorted(
        {
            row.get("comparison_group", "")
            for row in rows
            if row.get("equivalence_status") == "mismatch"
        }
        - {""}
    )
    return {
        "cases": len({row["case"] for row in rows}),
        "rows": len(rows),
        "algorithms": algorithms,
        "observed_pair_outcomes": {
            competitor: dict(
                sorted(observed_pair_outcomes(rows, competitor).items())
            )
            for competitor in ("fast", "incremental3")
        },
        "mismatch_cases": mismatch_cases,
        "equivalence_mismatches": equivalence_mismatches,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _cactus_panel(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, str]],
    *,
    timeout_s: float,
) -> None:
    minimum = timeout_s
    for algorithm in ALGORITHMS:
        times = sorted(
            float(row["solver_time_s"])
            for row in rows
            if row["algorithm"] == algorithm and row["status"] == "ok"
        )
        if times:
            minimum = min(minimum, times[0])
            axis.step(
                times,
                range(1, len(times) + 1),
                where="post",
                color=COLORS[algorithm],
                linewidth=1.8,
                label=f"{LABELS[algorithm]} ({len(times)}/165)",
            )
    axis.axvline(timeout_s, color="#666666", linestyle=(0, (3, 2)), linewidth=0.8)
    axis.text(
        timeout_s * 0.94,
        4,
        f"{timeout_s:g} s limit",
        color="#555555",
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    axis.set_xscale("log")
    axis.set_xlim(max(minimum / 1.5, 1e-5), timeout_s * 1.35)
    axis.set_ylim(0, 170)
    axis.set_xlabel("Median solve time (s, log scale)")
    axis.set_ylabel("Cases solved")
    axis.set_title("(a) Coverage across all 165 cases", loc="left", fontweight="bold")
    axis.grid(True, which="major", color="#D9D9D9", linewidth=0.55)
    axis.grid(True, which="minor", axis="x", color="#ECECEC", linewidth=0.4)
    axis.legend(loc="upper left", frameon=False, handlelength=2.3)


def _pair_outcome_panel(
    axis: plt.Axes,
    rows: Sequence[Mapping[str, str]],
) -> None:
    competitors = ("fast", "incremental3")
    outcome_order = (
        "both_solved",
        "bp_dp_only",
        "competitor_only",
        "neither_solved",
    )
    outcome_labels = {
        "both_solved": "Both solved",
        "bp_dp_only": "BP-DP only",
        "competitor_only": "Competitor only",
        "neither_solved": "Neither solved",
    }
    outcome_colors = {
        "both_solved": "#777777",
        "bp_dp_only": COLORS["boundary-profile"],
        "competitor_only": "#E69F00",
        "neither_solved": "#D9D9D9",
    }
    bottoms = [0, 0]
    for outcome in outcome_order:
        values = [
            observed_pair_outcomes(rows, competitor)[outcome]
            for competitor in competitors
        ]
        bars = axis.bar(
            range(len(competitors)),
            values,
            bottom=bottoms,
            width=0.58,
            color=outcome_colors[outcome],
            label=outcome_labels[outcome],
            edgecolor="white",
            linewidth=0.5,
        )
        for index, (bar, value) in enumerate(zip(bars, values)):
            if value:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[index] + value / 2,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=("white" if outcome != "neither_solved" else "#333333"),
                )
            bottoms[index] += value
    axis.set_xticks(range(len(competitors)))
    axis.set_xticklabels([LABELS[competitor] for competitor in competitors])
    axis.set_ylabel("Observed BP-DP/baseline pairs")
    axis.set_title("(a) Solve outcomes on observed pairs", loc="left", fontweight="bold")
    axis.set_ylim(0, max(bottoms, default=1) * 1.12)
    axis.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    axis.legend(loc="upper right", frameon=False, fontsize=6.8)


def _speedup_panel(axis: plt.Axes, rows: Sequence[Mapping[str, str]]) -> None:
    competitors = ("fast", "incremental3")
    offsets = {"fast": -0.18, "incremental3": 0.18}
    populated_categories: set[str] = set()
    for competitor in competitors:
        points = paired_speedups(rows, competitor)
        for category_index, category in enumerate(CATEGORIES):
            values = [value for point_category, value in points if point_category == category]
            if not values:
                continue
            populated_categories.add(category)
            position = category_index + offsets[competitor]
            box = axis.boxplot(
                values,
                positions=[position],
                widths=0.27,
                patch_artist=True,
                showfliers=False,
                whis=(10, 90),
                medianprops={"color": "white", "linewidth": 1.25},
                boxprops={"linewidth": 0.7},
                whiskerprops={"linewidth": 0.7},
                capprops={"linewidth": 0.7},
            )
            box["boxes"][0].set_facecolor(COLORS[competitor])
            box["boxes"][0].set_alpha(0.82)
            for index, value in enumerate(values):
                jitter = (((index * 37) % 19) - 9) / 440
                axis.scatter(
                    position + jitter,
                    value,
                    s=7,
                    color=COLORS[competitor],
                    alpha=0.34,
                    linewidths=0,
                    zorder=3,
                )
    axis.axhline(1, color="#333333", linestyle=(0, (3, 2)), linewidth=0.85)
    for category_index, category in enumerate(CATEGORIES):
        if category not in populated_categories:
            axis.text(
                category_index,
                1,
                "no paired\nsolves",
                ha="center",
                va="center",
                fontsize=6.8,
                color="#666666",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
                zorder=5,
            )
    axis.set_yscale("log")
    axis.set_xlim(-0.55, len(CATEGORIES) - 0.45)
    axis.set_xticks(range(len(CATEGORIES)))
    axis.set_xticklabels([CATEGORY_LABELS[category] for category in CATEGORIES])
    axis.set_ylabel("Competitor / BP-DP runtime")
    axis.set_title("(b) Paired speedup by category", loc="left", fontweight="bold")
    axis.grid(True, which="major", axis="y", color="#D9D9D9", linewidth=0.55)
    axis.grid(True, which="minor", axis="y", color="#ECECEC", linewidth=0.4)
    axis.text(
        0.99,
        0.965,
        "above 1: BP-DP is faster",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#444444",
    )
    axis.legend(
        handles=[
            Line2D([0], [0], color=COLORS[algorithm], lw=5, label=LABELS[algorithm])
            for algorithm in competitors
        ],
        loc="lower right",
        frameon=False,
    )


def plot_results(
    rows: Sequence[Mapping[str, str]],
    output: Path,
    *,
    timeout_s: float = 30,
    partial: bool = False,
) -> None:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(7.05, 3.05))
    if partial:
        _pair_outcome_panel(axes[0], rows)
    else:
        _cactus_panel(axes[0], rows, timeout_s=timeout_s)
    _speedup_panel(axes[1], rows)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout(w_pad=2.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def plot_scaling_results(
    rows: Sequence[Mapping[str, str]],
    output: Path,
    *,
    timeout_s: float = 30,
    partial: bool = False,
) -> None:
    """Plot the four runtime-by-domain comparisons used by FastWFOMC."""

    _style()
    figure, axes = plt.subplots(1, 4, figsize=(7.05, 2.45))
    panel_names = "abcd"
    for panel_index, (title, category, family, variant) in enumerate(
        SCALING_FAMILIES
    ):
        axis = axes[panel_index]
        selected = select_scaling_rows(
            rows,
            category=category,
            family=family,
            variant=variant,
        )
        successful = [
            float(row["solver_time_s"])
            for row in selected
            if row["status"] == "ok"
        ]
        minimum = min(successful, default=timeout_s)
        lower_limit = max(minimum / 1.8, 1e-4)
        domains = sorted({int(row["domain_size"]) for row in selected})
        domain_span = max(domains, default=1) - min(domains, default=0)
        offset_step = max(domain_span * 0.008, 0.12)
        x_offsets = {
            "boundary-profile": -offset_step,
            "fast": 0.0,
            "incremental3": offset_step,
        }
        axis.axhline(
            timeout_s,
            color="#777777",
            linestyle=(0, (3, 2)),
            linewidth=0.7,
            zorder=1,
        )
        for algorithm in ALGORITHMS:
            algorithm_rows = [
                row for row in selected if row["algorithm"] == algorithm
            ]
            solved = [row for row in algorithm_rows if row["status"] == "ok"]
            if solved:
                axis.plot(
                    [
                        int(row["domain_size"]) + x_offsets[algorithm]
                        for row in solved
                    ],
                    [float(row["solver_time_s"]) for row in solved],
                    color=COLORS[algorithm],
                    marker=MARKERS[algorithm],
                    markersize=3.4,
                    linewidth=1.35,
                    markeredgewidth=0.5,
                    label=LABELS[algorithm],
                    zorder=3,
                )
            for status, marker in (("timeout", "v"), ("memory", "X")):
                failures = [row for row in algorithm_rows if row["status"] == status]
                if failures:
                    axis.scatter(
                        [
                            int(row["domain_size"]) + x_offsets[algorithm]
                            for row in failures
                        ],
                        [timeout_s] * len(failures),
                        marker=marker,
                        s=18,
                        facecolors=("none" if status == "timeout" else COLORS[algorithm]),
                        edgecolors=COLORS[algorithm],
                        linewidths=0.8,
                        zorder=4,
                    )
            if partial:
                observed_domains = {
                    int(row["domain_size"]) for row in algorithm_rows
                }
                missing_domains = [
                    domain for domain in domains if domain not in observed_domains
                ]
                if missing_domains:
                    axis.scatter(
                        [
                            domain + x_offsets[algorithm]
                            for domain in missing_domains
                        ],
                        [lower_limit * 1.12] * len(missing_domains),
                        marker="d",
                        s=14,
                        facecolors="none",
                        edgecolors=COLORS[algorithm],
                        linewidths=0.7,
                        zorder=4,
                    )
        axis.set_yscale("log")
        axis.set_ylim(lower_limit, timeout_s * 1.65)
        axis.set_xlabel("Domain size $n$")
        if panel_index == 0:
            axis.set_ylabel("Median runtime (s)")
        axis.set_title(
            f"({panel_names[panel_index]}) {title}",
            loc="left",
            fontweight="bold",
            fontsize=8.2,
        )
        axis.grid(True, which="major", axis="y", color="#D9D9D9", linewidth=0.5)
        axis.grid(True, which="minor", axis="y", color="#EEEEEE", linewidth=0.35)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS[algorithm],
                marker=MARKERS[algorithm],
                linewidth=1.5,
                markersize=4,
                label=LABELS[algorithm],
            )
            for algorithm in ALGORITHMS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.4,
    )
    if partial:
        figure.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#666666",
                    marker="v",
                    markerfacecolor="none",
                    linestyle="none",
                    label="Timeout",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#666666",
                    marker="X",
                    linestyle="none",
                    label="Memory limit",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#666666",
                    marker="d",
                    markerfacecolor="none",
                    linestyle="none",
                    label="Not run",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.015),
            ncol=3,
            frameon=False,
            handletextpad=0.35,
            columnspacing=1.2,
        )
    figure.tight_layout(
        rect=(0, 0.09 if partial else 0, 1, 0.9),
        w_pad=0.75,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--scaling-figure", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=165)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="accept an interrupted run and plot only observed pair comparisons",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_results(args.results)
    validate_results(
        rows,
        expected_cases=args.expected_cases,
        allow_partial=args.allow_partial,
    )
    expected_rows = args.expected_cases * len(ALGORITHMS)
    is_complete = (
        len(rows) == expected_rows
        and len({row["case"] for row in rows}) == args.expected_cases
    )
    summary = summarize(rows)
    summary["expected_cases"] = args.expected_cases
    summary["expected_rows"] = expected_rows
    summary["complete"] = is_complete
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_results(
        rows,
        args.figure,
        timeout_s=args.timeout,
        partial=not is_complete,
    )
    if args.scaling_figure is not None:
        plot_scaling_results(
            rows,
            args.scaling_figure,
            timeout_s=args.timeout,
            partial=not is_complete,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
