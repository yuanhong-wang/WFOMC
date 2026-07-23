# Flat Benchmarks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the suite-selected benchmark collection with one flat 165-case catalog and one current runner.

**Architecture:** `benchmarks/cases.py` owns one immutable catalog and key lookup. `benchmarks/run.py` runs that full catalog directly, with no historical-source or suite abstraction; historical scripts and checked-in results are deleted.

**Tech Stack:** Python 3.11, argparse, pytest, existing WFOMC typed API.

---

### Task 1: Specify the flat catalog

**Files:**
- Modify: `tests/unit/test_benchmark_cases.py`
- Modify: `benchmarks/cases.py`

1. Replace suite-size tests with one 165-case inventory test.
2. Make the new test fail against the selector API.
3. Remove `_SUITES`, `benchmark_suite_names()`, and the `suite` parameter.
4. Run `uv run pytest -q tests/unit/test_benchmark_cases.py` and expect PASS.

### Task 2: Consolidate the current runner

**Files:**
- Create: `benchmarks/run.py`
- Delete: `benchmarks/domain_series_performance.py`
- Modify: `tests/unit/test_domain_series_performance.py`

1. Update tests to import `benchmarks.run` and operate on concrete cases.
2. Move the current runner to `run.py`.
3. Remove model workloads, serialized cross-branch workloads, `--suite`,
   `--sources`, and selector fields from manifests/results/summaries.
4. Keep cold/compile-once protocols, algorithms, resource limits, resumption,
   correctness, CSV, and summary output.
5. Run the runner unit tests and a one-case CLI smoke run.

### Task 3: Remove historical benchmark material

**Files:**
- Delete: `benchmarks/boundary_profile_performance.py`
- Delete: `benchmarks/cross_branch_performance.py`
- Delete: `benchmarks/cross_branch_worker.py`
- Delete: `benchmarks/results/`
- Delete: `tests/unit/test_boundary_profile_performance.py`
- Delete: `tests/unit/test_cross_branch_performance.py`

1. Delete the approved historical runners, tests, and tracked artifacts.
2. Remove generated caches and Finder metadata.
3. Search the active codebase for deleted module names and selector terminology.

### Task 4: Document and verify

**Files:**
- Modify: `benchmarks/README.md`

1. Document the three-file layout and selector-free CLI.
2. Run focused benchmark tests.
3. Run `uv run pytest -q` and expect all remaining tests to pass.
4. Run `git diff --check` and verify `find benchmarks -maxdepth 1 -type f`
   lists only `README.md`, `cases.py`, and `run.py`.
