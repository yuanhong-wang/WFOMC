from __future__ import annotations

import pytest

from benchmarks.cases import benchmark_case
from benchmarks.run import (
    ALGORITHMS,
    CaseGroup,
    _run_worker,
    build_manifest,
    group_cases,
    mark_correctness,
    paired_stats,
    parse_args,
    prepare_resume,
    run_stopping_series_process,
    write_manifest,
)


def test_runner_uses_the_three_paper_algorithms() -> None:
    assert ALGORITHMS == ("boundary-profile", "fast", "incremental3")


def test_group_cases_preserves_the_selected_inventory() -> None:
    cases = tuple(
        benchmark_case(key)
        for key in (
            "core/properly-4-coloured-graph/n100",
            "core/properly-4-coloured-graph/n200",
            "core/permutations/fo2-cardinality-reduction/n75",
        )
    )
    groups = group_cases(cases)

    grouped_keys = [case.key for group in groups for case in group.cases]
    assert sorted(grouped_keys) == sorted(case.key for case in cases)
    assert len(grouped_keys) == len(set(grouped_keys))


def test_paired_stats_reports_comparator_over_boundary_profile() -> None:
    rows = [
        {
            "case": case,
            "algorithm": algorithm,
            "status": "ok",
            "solver_time_s": seconds,
        }
        for case, algorithm, seconds in (
            ("a", "boundary-profile", 1.0),
            ("a", "fast", 4.0),
            ("b", "boundary-profile", 2.0),
            ("b", "fast", 1.0),
            ("c", "boundary-profile", 1.0),
            ("c", "fast", None),
        )
    ]
    rows[-1]["status"] = "timeout"

    assert paired_stats(rows, "fast") == {
        "pairs": 2,
        "geomean": pytest.approx(2**0.5),
        "median": 2.25,
        "boundary_faster_pct": 50.0,
    }


def test_worker_reuses_boundary_profile_template_across_domains() -> None:
    payload = _run_worker(
        (
            "core/properly-4-coloured-graph/n100",
            "core/properly-4-coloured-graph/n200",
        ),
        "boundary-profile",
        timeout_s=30,
        repetitions=1,
    )

    assert [row["status"] for row in payload["rows"]] == ["ok", "ok"]
    assert payload["template_misses"] == 1
    assert payload["template_hits"] == 1


def test_cold_stop_series_skips_larger_domains_after_timeout(monkeypatch) -> None:
    cases = tuple(
        benchmark_case(key)
        for key in (
            "core/3-edge-disjoint-perfect-matchings/"
            "fo2-cardinality-reduction/n10",
            "core/3-edge-disjoint-perfect-matchings/"
            "fo2-cardinality-reduction/n20",
            "core/3-edge-disjoint-perfect-matchings/"
            "fo2-cardinality-reduction/n30",
        )
    )
    statuses = iter(("ok", "timeout"))
    calls = []

    def fake_run(singleton, algorithm, **kwargs):
        calls.append(singleton.cases[0].domain_size)
        status = next(statuses)
        return (
            {
                "rows": [
                    {
                        "status": status,
                        "solver_time_s": 1.0 if status == "ok" else None,
                    }
                ],
                "template_hits": 0,
                "template_misses": 1,
            },
            1.0,
            1024,
        )

    monkeypatch.setattr("benchmarks.run.run_series_process", fake_run)
    payload, _wall_time, _peak_rss = run_stopping_series_process(
        CaseGroup(
            series="core/3-edge-disjoint-perfect-matchings/pilot",
            family="3-edge-disjoint-perfect-matchings",
            category="core",
            variant="fo2-cardinality-reduction",
            cases=cases,
        ),
        "fast",
        timeout_s=300,
        memory_bytes=1024**3,
        repetitions=1,
    )

    assert calls == [10, 20]
    assert [row["status"] for row in payload["rows"]] == [
        "ok",
        "timeout",
        "skipped-after-resource",
    ]


def test_cold_worker_does_not_reuse_runtime_between_repetitions() -> None:
    payload = _run_worker(
        ("core/permutations/fo2-cardinality-reduction/n75",),
        "boundary-profile",
        timeout_s=30,
        repetitions=2,
        protocol="cold",
    )

    assert payload["rows"][0]["status"] == "ok"
    assert len(payload["rows"][0]["solver_time_samples_s"]) == 2
    assert payload["rows"][0]["is_warm_domain"] is False


def test_incremental3_worker_runs_the_original_c2_sentence() -> None:
    payload = _run_worker(
        ("core/permutations/fo2-cardinality-reduction/n8",),
        "incremental3",
        timeout_s=30,
        repetitions=1,
        protocol="cold",
    )
    row = payload["rows"][0]

    assert row["status"] == "ok"
    assert row["result"] == "40320"
    assert row["input_variant"] == "original-c2"
    assert row["correction_divisor"] == 1


def test_manifest_run_id_is_deterministic_and_covers_measurement_options() -> None:
    case = benchmark_case("core/properly-2-coloured-graph/n8")
    first = build_manifest(
        [case], protocol="cold", algorithms=("fast",), commit="abc",
        timeout_s=30, memory_bytes=1024,
        repetitions=3, environment={"python": "3.11"}, dirty_sha256="clean",
        lock_sha256="lock",
    )
    same = build_manifest(
        [case], protocol="cold", algorithms=("fast",), commit="abc",
        timeout_s=30, memory_bytes=1024,
        repetitions=3, environment={"python": "3.11"}, dirty_sha256="clean",
        lock_sha256="lock",
    )
    changed = build_manifest(
        [case], protocol="compile-once", algorithms=("fast",), commit="abc",
        timeout_s=30, memory_bytes=1024,
        repetitions=3, environment={"python": "3.11"}, dirty_sha256="clean",
        lock_sha256="lock",
    )
    changed_order = build_manifest(
        [case], protocol="cold", algorithms=("fast",), commit="abc",
        timeout_s=30, memory_bytes=1024,
        repetitions=3, environment={"python": "3.11"}, dirty_sha256="clean",
        lock_sha256="lock", order_seed=1,
    )

    assert first == same
    assert "suite" not in first
    assert "sources" not in first
    assert first["run_id"] != changed["run_id"]
    assert first["run_id"] != changed_order["run_id"]


def test_manifest_records_algorithm_specific_core_inputs() -> None:
    case = benchmark_case(
        "core/undirected-3-regular/fo2-cardinality-reduction/n30"
    )
    manifest = build_manifest(
        [case],
        protocol="cold",
        algorithms=("boundary-profile", "incremental3"),
        commit="abc",
        timeout_s=30,
        memory_bytes=1024,
        repetitions=1,
        environment={"python": "3.11"},
        dirty_sha256="clean",
        lock_sha256="lock",
    )
    inputs = manifest["cases"][0]["inputs"]

    assert inputs["boundary-profile"]["variant"] == "fo2-cardinality-reduction"
    assert inputs["boundary-profile"]["correction_divisor"] == 6**30
    assert inputs["incremental3"]["variant"] == "original-c2"
    assert inputs["incremental3"]["correction_divisor"] == 1
    assert (
        inputs["boundary-profile"]["problem"]
        != inputs["incremental3"]["problem"]
    )


def test_cli_has_no_inventory_selectors() -> None:
    args = parse_args([])

    assert not hasattr(args, "suite")
    assert not hasattr(args, "sources")
    assert not hasattr(args, "limit_workloads")


def test_resume_refuses_missing_or_mismatched_manifest(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "results.csv").write_text("case,status\na,ok\n")
    manifest = {"schema_version": 1, "run_id": "new"}

    with pytest.raises(ValueError, match="manifest"):
        prepare_resume(output, manifest, no_resume=False)

    write_manifest({"schema_version": 1, "run_id": "old"}, output / "manifest.json")
    with pytest.raises(ValueError, match="does not match"):
        prepare_resume(output, manifest, no_resume=False)

    assert prepare_resume(output, manifest, no_resume=True) == []


def test_correctness_requires_overlap_and_normalizes_comparison_groups() -> None:
    rows = [
        {
            "case": "singleton", "algorithm": "boundary-profile", "status": "ok",
            "result": "7", "comparison_group": "", "correction_divisor": 1,
        },
        {
            "case": "same", "algorithm": "boundary-profile", "status": "ok",
            "result": "9", "comparison_group": "", "correction_divisor": 1,
        },
        {
            "case": "same", "algorithm": "fast", "status": "ok",
            "result": "9", "comparison_group": "", "correction_divisor": 1,
        },
        {
            "case": "normalized", "algorithm": "boundary-profile", "status": "ok",
            "result": "30", "comparison_group": "", "correction_divisor": 6,
        },
        {
            "case": "normalized", "algorithm": "incremental3", "status": "ok",
            "result": "5", "comparison_group": "", "correction_divisor": 1,
        },
        {
            "case": "direct", "algorithm": "boundary-profile", "status": "ok",
            "result": "5", "comparison_group": "equivalent", "correction_divisor": 1,
        },
        {
            "case": "reduced", "algorithm": "boundary-profile", "status": "ok",
            "result": "30", "comparison_group": "equivalent", "correction_divisor": 6,
        },
    ]

    mark_correctness(rows)

    by_case = {row["case"]: row for row in rows}
    assert by_case["singleton"]["comparison_status"] == "not-comparable"
    assert by_case["same"]["comparison_status"] == "match"
    assert by_case["normalized"]["comparison_status"] == "match"
    assert by_case["direct"]["equivalence_status"] == "match"
    assert by_case["reduced"]["equivalence_status"] == "match"
