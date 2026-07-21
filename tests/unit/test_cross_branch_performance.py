from __future__ import annotations

import json

import pytest

import benchmarks.cross_branch_performance as cross_branch
from benchmarks.cases import benchmark_case
from benchmarks.cross_branch_performance import (
    RESOURCE_FAILURES,
    classify_error,
    collect_catalog_workloads,
    collect_model_workloads,
    parse_worker_output,
    paired_branch_stats,
    mark_consensus,
    rewrite_integer_domain,
    series_key,
    Workload,
    serialize_catalog_case,
)
from wfomc import parse_problem


def test_rewrite_integer_domain_preserves_surrounding_text() -> None:
    source = "\\forall X: P(X)\n\npeople = 7 # original\n1 -1 P\n"

    rewritten = rewrite_integer_domain(source, 32)

    assert rewritten == "\\forall X: P(X)\n\npeople = 32 # original\n1 -1 P\n"


def test_rewrite_integer_domain_rejects_named_domain() -> None:
    with pytest.raises(ValueError, match="integer domain"):
        rewrite_integer_domain("\\forall X: P(X)\nD = {alice, bob}\n", 8)


@pytest.mark.parametrize(
    "key",
    [
        "core/bi-total-relation/n75",
        "c2/undirected-3-regular/direct-c2/n10",
        "cardinality/properly-4-coloured-undirected-3-regular/"
        "fo2-cardinality-reduction/n10",
        "unary/bi-total-relation/sx-cardinality/exact/n20",
    ],
)
def test_serialized_catalog_case_round_trips_in_current_parser(key: str) -> None:
    case = benchmark_case(key)

    source = serialize_catalog_case(case)
    parsed = parse_problem(source)

    assert len(parsed.domain) == case.domain_size
    assert (
        parsed.problem.declared_predicate_names()
        == case.build_problem().problem.declared_predicate_names()
    )


def test_parse_worker_output_uses_sentinel_and_ignores_logs() -> None:
    payload = {"status": "ok", "result": "42", "solver_time_s": 0.125}
    stdout = "debug line\nBENCH_RESULT_JSON=" + json.dumps(payload) + "\n"

    assert parse_worker_output(stdout) == payload


def test_parse_worker_output_rejects_missing_sentinel() -> None:
    with pytest.raises(ValueError, match="sentinel"):
        parse_worker_output("ordinary output only")


def test_paired_branch_stats_uses_successful_matching_algorithm_pairs() -> None:
    rows = [
        {
            "source_sha256": "a",
            "domain_size": 2,
            "branch": branch,
            "algorithm": "fastv2",
            "status": "ok",
            "solver_time_s": time,
        }
        for branch, time in (("devel", 2.0), ("modk", 8.0))
    ]
    rows.append(
        {
            "source_sha256": "b",
            "domain_size": 4,
            "branch": "devel",
            "algorithm": "fastv2",
            "status": "ok",
            "solver_time_s": 1.0,
        }
    )

    assert paired_branch_stats(rows, "fastv2") == {
        "pairs": 1,
        "geomean": pytest.approx(4.0),
        "median": 4.0,
        "devel_faster_pct": 100.0,
    }


def test_resource_failures_block_the_whole_problem_series() -> None:
    row = {
        "source_kind": "catalog",
        "family": "undirected-3-regular",
        "variant": "default",
        "branch": "devel",
        "algorithm": "incremental3",
        "status": "timeout",
    }

    assert row["status"] in RESOURCE_FAILURES
    assert series_key(row) == ("catalog", "undirected-3-regular", "default")


def test_series_hash_ignores_only_the_integer_domain_declaration() -> None:
    small = Workload(
        "model", "demo/n2", "demo", "model", "default", 2,
        ".wfomcs", "P(X)\nD = 2\n|P| = 1\n",
    )
    large = Workload(
        "model", "demo/n4", "demo", "model", "default", 4,
        ".wfomcs", "P(X)\nD = 4\n|P| = 1\n",
    )
    changed_constraint = Workload(
        "model", "demo/n4", "demo", "model", "default", 4,
        ".wfomcs", "P(X)\nD = 4\n|P| = 2\n",
    )

    assert small.series_sha256 == large.series_sha256
    assert small.series_sha256 != changed_constraint.series_sha256
    assert series_key({"source_kind": "model", "series_sha256": small.series_sha256}) == (
        "model", small.series_sha256,
    )


def test_catalog_workload_preserves_equivalence_metadata() -> None:
    workloads = {row.case: row for row in collect_catalog_workloads("c2")}
    row = workloads[
        "c2/undirected-3-regular/fo2-cardinality-reduction/n10"
    ]

    assert row.comparison_group == "undirected-3-regular/n10"
    assert row.correction_divisor == 6**10


def test_model_source_is_read_from_selected_commit(monkeypatch) -> None:
    monkeypatch.setattr(
        cross_branch,
        "_tree_models",
        lambda commit: {"models/demo.wfomcs"},
    )
    calls = []

    def fake_git_show(commit, relative):
        calls.append((commit, relative))
        return "P(X)\nD = 7\n"

    monkeypatch.setattr(cross_branch, "_git_show", fake_git_show)

    workloads, exclusions = collect_model_workloads("current123", "base456")

    assert not exclusions
    assert workloads[0].source == "P(X)\nD = 2\n"
    assert calls == [("current123", "models/demo.wfomcs")]


def test_existing_baseline_worktree_still_syncs_locked_environment(
    tmp_path, monkeypatch
) -> None:
    commit = "a" * 40
    path = tmp_path / f"wfomc-cross-branch-{commit[:12]}"
    python = path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    calls = []

    monkeypatch.setattr(cross_branch, "resolve_commit", lambda _ref: commit)
    monkeypatch.setattr(cross_branch.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cross_branch, "_git", lambda *_args, **_kwargs: commit)
    monkeypatch.setattr(
        cross_branch.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    actual_path, actual_commit = cross_branch.prepare_modk_worktree("modk")

    assert (actual_path, actual_commit) == (path, commit)
    assert calls == [
        (("uv", "sync", "--frozen", "--project", str(path)), {"check": True})
    ]


def test_changed_domain_dependent_constraint_is_not_skipped(
    tmp_path, monkeypatch
) -> None:
    calls = 0

    def fake_run_worker(*args, **kwargs):
        nonlocal calls
        calls += 1
        status = "timeout" if calls == 1 else "ok"
        return {
            "status": status, "result": None if status == "timeout" else "1",
            "solver_time_s": None if status == "timeout" else 0.01,
            "wall_time_s": 0.01, "peak_rss_bytes": 1024,
            "peak_rss_mib": 1 / 1024, "error": "limit", "stderr": "",
        }

    monkeypatch.setattr(cross_branch, "run_worker", fake_run_worker)
    workloads = [
        Workload(
            "model", f"demo/n{n}", "demo", "model", "default", n,
            ".wfomcs", f"P(X)\nD = {n}\n|P| = {rhs}\n",
        )
        for n, rhs in ((2, 1), (4, 2))
    ]

    rows = cross_branch.execute(
        workloads, {"devel": (tmp_path / "python", "a")},
        timeout_s=30, memory_bytes=4 * 1024**3, repetitions=1,
        input_dir=tmp_path,
    )

    assert calls == 4
    assert [row["status"] for row in rows[2:]] == ["ok", "ok"]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("UnsupportedFeatureError: fastv2 does not support mod counting", "unsupported"),
        ("RuntimeError: Linear order axiom is only supported by incremental3", "unsupported"),
        ("ValueError: Counting sections not reducible to UFO2 + cardinality", "unsupported"),
        ("ValueError: Evidence must be consistent with the domain: person2", "invalid"),
        ("RuntimeError: internal invariant failed", "error"),
    ],
)
def test_classify_error_separates_expected_statuses(error, expected) -> None:
    assert classify_error(error) == expected


def test_resource_failure_skips_only_same_configuration_at_larger_domains(
    tmp_path, monkeypatch
) -> None:
    calls = 0

    def fake_run_worker(*args, **kwargs):
        nonlocal calls
        calls += 1
        status = "timeout" if calls == 1 else "ok"
        return {
            "status": status,
            "result": "1" if status == "ok" else None,
            "solver_time_s": 0.01 if status == "ok" else None,
            "wall_time_s": 0.01,
            "peak_rss_bytes": 1024,
            "peak_rss_mib": 1 / 1024,
            "error": "limit" if status == "timeout" else None,
            "stderr": "",
        }

    monkeypatch.setattr(cross_branch, "run_worker", fake_run_worker)
    workloads = [
        Workload("model", f"demo/n{n}", "demo", "model", "default", n, ".wfomcs", f"P(X)\nD = {n}\n")
        for n in (2, 4)
    ]

    rows = cross_branch.execute(
        workloads,
        {"devel": (tmp_path / "python", "a"), "modk": (tmp_path / "python", "b")},
        timeout_s=30,
        memory_bytes=4 * 1024**3,
        repetitions=1,
        input_dir=tmp_path,
    )

    assert calls == 7
    assert [row["status"] for row in rows[4:]] == [
        "skipped-after-resource", "ok", "ok", "ok"
    ]


def test_resume_requires_matching_measurement_identity(tmp_path, monkeypatch) -> None:
    calls = 0

    def fake_run_worker(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "status": "ok", "result": "1", "solver_time_s": 0.02,
            "wall_time_s": 0.03, "peak_rss_bytes": 1024,
            "peak_rss_mib": 1 / 1024, "error": "", "stderr": "",
        }

    monkeypatch.setattr(cross_branch, "run_worker", fake_run_worker)
    workload = Workload(
        "model", "demo/n2", "demo", "model", "default", 2,
        ".wfomcs", "P(X)\nD = 2\n",
    )
    stale = {
        "source_sha256": workload.source_sha256,
        "domain_size": "2",
        "branch": "devel",
        "algorithm": "fastv2",
        "commit": "old",
        "repetitions": "1",
        "timeout_s": "30",
        "memory_bytes": str(4 * 1024**3),
        "run_id": "run",
        "status": "ok",
        "solver_time_s": "999",
    }

    rows = cross_branch.execute(
        [workload], {"devel": (tmp_path / "python", "new")},
        timeout_s=30, memory_bytes=4 * 1024**3, repetitions=1,
        input_dir=tmp_path, resume_rows=[stale], run_id="run",
    )

    assert calls == 2
    assert {row["solver_time_s"] for row in rows} == {0.02}


def test_result_schema_preserves_timing_distribution() -> None:
    assert "solver_time_min_s" in cross_branch.RESULT_FIELDS
    assert "solver_time_max_s" in cross_branch.RESULT_FIELDS
    assert "solver_time_samples_s" in cross_branch.RESULT_FIELDS


def test_consensus_requires_two_successes_and_checks_equivalent_encodings() -> None:
    rows = [
        {
            "source_sha256": "single", "domain_size": 2, "case": "single",
            "branch": "devel", "algorithm": "fastv2", "status": "ok",
            "result": "7", "comparison_group": "", "correction_divisor": 1,
        },
        {
            "source_sha256": "shared", "domain_size": 2, "case": "shared",
            "branch": "devel", "algorithm": "fastv2", "status": "ok",
            "result": "9", "comparison_group": "", "correction_divisor": 1,
        },
        {
            "source_sha256": "shared", "domain_size": 2, "case": "shared",
            "branch": "modk", "algorithm": "fastv2", "status": "ok",
            "result": "9", "comparison_group": "", "correction_divisor": 1,
        },
        {
            "source_sha256": "direct", "domain_size": 10, "case": "direct",
            "branch": "devel", "algorithm": "fastv2", "status": "ok",
            "result": "5", "comparison_group": "equiv", "correction_divisor": 1,
        },
        {
            "source_sha256": "reduced", "domain_size": 10, "case": "reduced",
            "branch": "devel", "algorithm": "fastv2", "status": "ok",
            "result": "30", "comparison_group": "equiv", "correction_divisor": 6,
        },
    ]

    mark_consensus(rows)

    assert rows[0]["comparison_status"] == "not-comparable"
    assert rows[0]["matches_consensus"] is None
    assert rows[1]["comparison_status"] == rows[2]["comparison_status"] == "match"
    assert rows[3]["equivalence_status"] == rows[4]["equivalence_status"] == "match"
