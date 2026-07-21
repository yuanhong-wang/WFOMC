# Unified Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build one reproducible current-algorithm benchmark with cold and compile-once protocols, and repair correctness-critical behavior in the legacy cross-branch runner.

**Architecture:** Extend the current domain-series runner around a shared workload and manifest model, using isolated worker processes for resource enforcement.  Keep the legacy branch worker for API compatibility, but give it exact source identity, safe resume rules, per-configuration truncation, and richer result fields.

**Tech Stack:** Python 3.11, argparse, dataclasses, psutil, pytest, existing WFOMC public API.

---

### Task 1: Shared workload identity and status model

**Files:**
- Modify: `benchmarks/cross_branch_performance.py`
- Test: `tests/unit/test_cross_branch_performance.py`

1. Add failing tests for domain-free series hashes, commit-sourced model text,
   and error classification.
2. Run the focused tests and confirm failure.
3. Add `series_sha256`, catalog comparison metadata, commit source loading, and
   explicit `unsupported`/`invalid` classifiers.
4. Run the focused tests and confirm success.

### Task 2: Safe resume and independent truncation

**Files:**
- Modify: `benchmarks/cross_branch_performance.py`
- Modify: `benchmarks/boundary_profile_performance.py`
- Test: `tests/unit/test_cross_branch_performance.py`
- Test: `tests/unit/test_boundary_profile_performance.py`

1. Add failing tests proving stale commits/options are not resumable and one
   configuration cannot skip another.
2. Add an explicit run identity to rows and scope blocked series by
   branch/algorithm/domain-free problem.
3. Require matching commit, repetitions, timeout, memory limit, and run id for
   resume in both runners.
4. Run both focused test modules.

### Task 3: Per-repetition limits and complete measurements

**Files:**
- Modify: `benchmarks/cross_branch_worker.py`
- Modify: `benchmarks/cross_branch_performance.py`
- Test: `tests/unit/test_cross_branch_performance.py`

1. Add tests for retained min/max/sample timings and per-repetition timeout
   metadata.
2. Add a worker-side per-solve deadline and preserve timing samples in CSV.
3. Keep the parent process deadline as a crash/native-call safety bound.
4. Run focused worker and runner tests.

### Task 4: Canonical cold and compile-once runner

**Files:**
- Modify: `benchmarks/domain_series_performance.py`
- Test: `tests/unit/test_domain_series_performance.py`

1. Add tests for `--protocol cold|compile-once`, catalog/model inventories,
   exact problem grouping, and series-local cache reuse.
2. Generalize workers to consume catalog or serialized model workloads.
3. Add independent algorithm truncation and preserve first-domain, warm-domain,
   compile, and series-total timings.
4. Run protocol smoke tests for all three algorithms.

### Task 5: Manifest and environment reproducibility

**Files:**
- Modify: `benchmarks/domain_series_performance.py`
- Test: `tests/unit/test_domain_series_performance.py`

1. Add failing tests for deterministic run ids and manifest mismatch refusal.
2. Record algorithm/protocol/options, workload digest, commit, dirty hash,
   platform/Python information, and lockfile hash.
3. Add exact-match resume and atomic manifest/result writes.
4. Verify interruption and resume using a smoke output directory.

### Task 6: Correctness and summaries

**Files:**
- Modify: `benchmarks/domain_series_performance.py`
- Modify: `benchmarks/cross_branch_performance.py`
- Test: `tests/unit/test_domain_series_performance.py`
- Test: `tests/unit/test_cross_branch_performance.py`

1. Add tests for singleton `not-comparable`, same-input mismatch, and normalized
   `comparison_group` equality.
2. Implement correctness states and correction-divisor normalization.
3. Separate unsupported, invalid, timeout, memory, and solver errors in reports.
4. Add solved-under-budget and stratified paired-success summaries.

### Task 7: Documentation and full verification

**Files:**
- Modify: `benchmarks/boundary_profile_performance.py`
- Modify: benchmark result documentation as needed

1. Mark saved-result BP comparisons as historical.
2. Document the canonical CLI and protocol meanings.
3. Run benchmark unit tests, compileall, diff checks, and the full pytest suite.
4. Run cold and compile-once smoke benchmarks and inspect manifests/results.
