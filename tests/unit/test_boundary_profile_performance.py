from pathlib import Path

import benchmarks.boundary_profile_performance as benchmark
from benchmarks.cross_branch_performance import Workload


def _workload(family: str, domain: int) -> Workload:
    return Workload(
        "model", f"{family}/n{domain}", family, "model", "default", domain,
        ".wfomcs", f"{family}(X)\nD = {domain}\n",
    )


def test_resource_failure_skips_larger_domains_in_same_series(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_run_worker(python, path, algorithm, **kwargs):
        calls.append(path.read_text())
        status = "timeout" if "D = 2" in calls[-1] else "ok"
        return {
            "status": status,
            "result": None if status == "timeout" else "1",
            "solver_time_s": None if status == "timeout" else 0.01,
            "wall_time_s": 0.01,
            "peak_rss_bytes": 1024,
            "peak_rss_mib": 1 / 1024,
            "error": "limit" if status == "timeout" else None,
            "stderr": "",
        }

    monkeypatch.setattr(benchmark, "run_worker", fake_run_worker)
    rows = benchmark.execute_boundary_profile(
        [_workload("a", 2), _workload("a", 4), _workload("b", 4)],
        python=Path("python"), commit="abc", timeout_s=30,
        memory_bytes=4 * 1024**3, repetitions=1, input_dir=tmp_path,
    )

    assert [row["status"] for row in rows] == [
        "timeout", "skipped-after-resource", "ok"
    ]
    assert len(calls) == 2


def test_saved_consensus_marks_only_comparable_successes() -> None:
    boundary_rows = [
        {"source_sha256": "a", "domain_size": 2, "status": "ok", "result": "7"},
        {"source_sha256": "b", "domain_size": 2, "status": "ok", "result": "9"},
    ]
    saved = [
        {"source_sha256": "a", "domain_size": "2", "status": "ok", "result": "7"},
    ]

    benchmark.mark_against_saved_consensus(boundary_rows, saved)

    assert boundary_rows[0]["matches_consensus"] is True
    assert boundary_rows[1]["matches_consensus"] is None


def test_saved_consensus_rejects_ambiguous_historical_results() -> None:
    boundary_rows = [
        {"source_sha256": "a", "domain_size": 2, "status": "ok", "result": "7"},
    ]
    saved = [
        {"source_sha256": "a", "domain_size": "2", "status": "ok", "result": value}
        for value in ("7", "8")
    ]

    benchmark.mark_against_saved_consensus(boundary_rows, saved)

    assert boundary_rows[0]["matches_consensus"] is None


def test_resume_rejects_changed_measurement_options(tmp_path, monkeypatch) -> None:
    calls = 0

    def fake_run_worker(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            "status": "ok", "result": "1", "solver_time_s": 0.01,
            "wall_time_s": 0.01, "peak_rss_bytes": 1024,
            "peak_rss_mib": 1 / 1024, "error": "", "stderr": "",
        }

    monkeypatch.setattr(benchmark, "run_worker", fake_run_worker)
    workload = _workload("a", 2)
    stale = {
        "source_sha256": workload.source_sha256, "domain_size": "2",
        "commit": "abc", "algorithm": "boundary-profile", "status": "ok",
        "repetitions": "1", "timeout_s": "10",
        "memory_bytes": str(4 * 1024**3), "run_id": "run",
        "solver_time_s": "999",
    }

    rows = benchmark.execute_boundary_profile(
        [workload], python=Path("python"), commit="abc", timeout_s=30,
        memory_bytes=4 * 1024**3, repetitions=1, input_dir=tmp_path,
        resume_rows=[stale], run_id="run",
    )

    assert calls == 1
    assert rows[0]["solver_time_s"] == 0.01
