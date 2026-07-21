from __future__ import annotations

import json

import pytest

import benchmarks.cross_branch_performance as cross_branch
from benchmarks.cases import benchmark_case
from benchmarks.cross_branch_performance import (
    RESOURCE_FAILURES,
    parse_worker_output,
    paired_branch_stats,
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
        "core/row-column/n75",
        "c2/3-regular/n10",
        "cardinality/3-regular-properly-4-coloured/n10",
        "unary/row-column-Sx/exact/n20",
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
        "family": "3-regular",
        "variant": "default",
        "branch": "devel",
        "algorithm": "incremental3",
        "status": "timeout",
    }

    assert row["status"] in RESOURCE_FAILURES
    assert series_key(row) == ("catalog", "3-regular", "default")


def test_any_resource_failure_skips_all_configurations_at_larger_domains(
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

    assert calls == 4  # all configurations still run at the boundary domain
    assert [row["status"] for row in rows[4:]] == ["skipped-after-resource"] * 4
