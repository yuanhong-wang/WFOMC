from __future__ import annotations

import pytest

from scripts.plot_benchmark_results import (
    observed_pair_outcomes,
    select_scaling_rows,
    summarize,
    validate_results,
)


def _row(case, category, algorithm, seconds, status="ok"):
    return {
        "case": case,
        "category": category,
        "algorithm": algorithm,
        "status": status,
        "solver_time_s": "" if seconds is None else str(seconds),
        "comparison_status": "match" if status == "ok" else "not-comparable",
        "equivalence_status": "not-applicable",
        "comparison_group": "",
    }


def test_summary_preserves_failures_and_uses_only_paired_successes() -> None:
    rows = [
        _row("a", "core", "boundary-profile", 1),
        _row("a", "core", "fast", 4),
        _row("a", "core", "incremental3", 2),
        _row("b", "cardinality", "boundary-profile", 2),
        _row("b", "cardinality", "fast", 1),
        _row("b", "cardinality", "incremental3", None, "timeout"),
    ]

    validate_results(rows, expected_cases=2)
    summary = summarize(rows)

    assert summary["algorithms"]["incremental3"]["status_counts"] == {
        "ok": 1,
        "timeout": 1,
    }
    assert summary["algorithms"]["fast"]["paired_with_bp_dp"] == {
        "pairs": 2,
        "geometric_mean_competitor_over_bp_dp": pytest.approx(2**0.5),
        "median_competitor_over_bp_dp": 2.25,
        "bp_dp_faster_cases": 1,
        "competitor_faster_cases": 1,
        "ties": 0,
        "by_category": {
            "core": {
                "pairs": 1,
                "geometric_mean_competitor_over_bp_dp": 4,
            },
            "cardinality": {
                "pairs": 1,
                "geometric_mean_competitor_over_bp_dp": 0.5,
            },
        },
    }


def test_scaling_selection_uses_category_family_and_variant() -> None:
    rows = [
        {
            **_row("selected", "c2", "boundary-profile", 1),
            "family": "undirected-3-regular",
            "variant": "fo2-cardinality-reduction",
            "domain_size": "20",
        },
        {
            **_row("wrong-category", "core", "boundary-profile", 1),
            "family": "undirected-3-regular",
            "variant": "fo2-cardinality-reduction",
            "domain_size": "10",
        },
        {
            **_row("wrong-variant", "c2", "boundary-profile", 1),
            "family": "undirected-3-regular",
            "variant": "direct-c2",
            "domain_size": "10",
        },
    ]

    selected = select_scaling_rows(
        rows,
        category="c2",
        family="undirected-3-regular",
        variant="fo2-cardinality-reduction",
    )

    assert [row["case"] for row in selected] == ["selected"]


def test_partial_validation_and_observed_pair_outcomes() -> None:
    rows = [
        _row("a", "core", "boundary-profile", 1),
        _row("a", "core", "fast", 2),
        _row("b", "core", "boundary-profile", 1),
        _row("b", "core", "incremental3", None, "timeout"),
        _row("c", "core", "fast", 1),
        _row("c", "core", "incremental3", 1),
    ]

    validate_results(rows, expected_cases=3, allow_partial=True)

    assert observed_pair_outcomes(rows, "fast") == {"both_solved": 1}
    assert observed_pair_outcomes(rows, "incremental3") == {"bp_dp_only": 1}

    with pytest.raises(ValueError, match="expected 9 rows"):
        validate_results(rows, expected_cases=3)
