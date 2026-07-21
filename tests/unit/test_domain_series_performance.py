from __future__ import annotations

import pytest

import benchmarks.domain_series_performance as benchmark
from benchmarks.cases import benchmark_cases
from benchmarks.cross_branch_performance import Workload
from benchmarks.domain_series_performance import (
    _run_worker,
    build_manifest,
    collect_current_model_workloads,
    group_cases,
    group_workloads,
    mark_correctness,
    paired_stats,
    prepare_resume,
    write_manifest,
)


def test_group_cases_reuses_one_problem_per_core_main_series() -> None:
    groups = group_cases(benchmark_cases("core-main"))

    assert {
        (group.family, tuple(case.domain_size for case in group.cases))
        for group in groups
    } == {
        ("bi-total-relation", (75, 100, 150)),
        ("3-neighbour-surjection-kernel", (30, 60, 100)),
        ("properly-4-coloured-graph", (100, 200, 300)),
        ("loopless-bi-total-relation", (100, 200, 300)),
        ("3-edge-disjoint-edge-covers", (10, 20, 30, 40)),
    }


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
            ("a", "fastv2", 4.0),
            ("b", "boundary-profile", 2.0),
            ("b", "fastv2", 1.0),
            ("c", "boundary-profile", 1.0),
            ("c", "fastv2", None),
        )
    ]
    rows[-1]["status"] = "timeout"

    assert paired_stats(rows, "fastv2") == {
        "pairs": 2,
        "geomean": pytest.approx(2**0.5),
        "median": 2.25,
        "boundary_faster_pct": 50.0,
    }


def test_worker_reuses_boundary_profile_template_across_domains() -> None:
    payload = _run_worker(
        ("core/bi-total-relation/n75", "core/bi-total-relation/n100"),
        "boundary-profile",
        timeout_s=30,
        repetitions=1,
    )

    assert [row["status"] for row in payload["rows"]] == ["ok", "ok"]
    assert payload["template_misses"] == 1
    assert payload["template_hits"] == 1


def test_cold_worker_does_not_reuse_runtime_between_repetitions() -> None:
    payload = _run_worker(
        ("core/bi-total-relation/n75",),
        "boundary-profile",
        timeout_s=30,
        repetitions=2,
        protocol="cold",
    )

    assert payload["rows"][0]["status"] == "ok"
    assert len(payload["rows"][0]["solver_time_samples_s"]) == 2
    assert payload["rows"][0]["is_warm_domain"] is False


def test_group_workloads_splits_domain_dependent_problem_changes() -> None:
    stable = [
        Workload(
            "model", f"stable/n{n}", "stable", "model", "default", n,
            ".wfomcs", f"P(X)\nD = {n}\n|P| = 1\n",
        )
        for n in (2, 4)
    ]
    changing = [
        Workload(
            "model", f"changing/n{n}", "changing", "model", "default", n,
            ".wfomcs", f"Q(X)\nD = {n}\n|Q| = {n // 2}\n",
        )
        for n in (2, 4)
    ]

    groups = group_workloads((*stable, *changing))

    assert sorted(len(group.workloads) for group in groups) == [1, 1, 2]


def test_current_model_inventory_reads_selected_commit(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark, "_tree_models", lambda commit: {"models/demo.wfomcs"}
    )
    calls = []

    def fake_show(commit, relative):
        calls.append((commit, relative))
        return "P(X)\nD = 99\n"

    monkeypatch.setattr(benchmark, "_git_show", fake_show)

    workloads, exclusions = collect_current_model_workloads("abc", (2, 4))

    assert not exclusions
    assert [workload.source for workload in workloads] == [
        "P(X)\nD = 2\n", "P(X)\nD = 4\n"
    ]
    assert calls == [("abc", "models/demo.wfomcs")]


def test_manifest_run_id_is_deterministic_and_covers_measurement_options() -> None:
    workload = Workload(
        "model", "demo/n2", "demo", "model", "default", 2,
        ".wfomcs", "P(X)\nD = 2\n",
    )
    first = build_manifest(
        [workload], protocol="cold", algorithms=("fastv2",), suite="all",
        sources="models", commit="abc", timeout_s=30, memory_bytes=1024,
        repetitions=3, environment={"python": "3.11"}, dirty_sha256="clean",
        lock_sha256="lock",
    )
    same = build_manifest(
        [workload], protocol="cold", algorithms=("fastv2",), suite="all",
        sources="models", commit="abc", timeout_s=30, memory_bytes=1024,
        repetitions=3, environment={"python": "3.11"}, dirty_sha256="clean",
        lock_sha256="lock",
    )
    changed = build_manifest(
        [workload], protocol="compile-once", algorithms=("fastv2",), suite="all",
        sources="models", commit="abc", timeout_s=30, memory_bytes=1024,
        repetitions=3, environment={"python": "3.11"}, dirty_sha256="clean",
        lock_sha256="lock",
    )
    changed_order = build_manifest(
        [workload], protocol="cold", algorithms=("fastv2",), suite="all",
        sources="models", commit="abc", timeout_s=30, memory_bytes=1024,
        repetitions=3, environment={"python": "3.11"}, dirty_sha256="clean",
        lock_sha256="lock", order_seed=1,
    )

    assert first == same
    assert first["run_id"] != changed["run_id"]
    assert first["run_id"] != changed_order["run_id"]


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
            "case": "same", "algorithm": "fastv2", "status": "ok",
            "result": "9", "comparison_group": "", "correction_divisor": 1,
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
    assert by_case["direct"]["equivalence_status"] == "match"
    assert by_case["reduced"]["equivalence_status"] == "match"
