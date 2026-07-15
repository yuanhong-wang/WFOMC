# Cross-Branch Performance Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and run a resource-bounded benchmark comparing `devel` and
`modk` implementations of `fastv2` and `incremental3` over growing domains.

**Architecture:** A branch-neutral controller serializes common workloads,
starts one isolated worker per measurement, monitors process-tree resources,
and creates CSV/Markdown/PNG reports.  A worker-side compatibility adapter
uses the API belonging to the checked-out commit.

**Tech Stack:** Python 3.11, git worktrees, uv, psutil, matplotlib, pytest.

---

### Task 1: Specify portable workloads and controller behavior

**Files:**
- Create: `tests/unit/test_cross_branch_performance.py`
- Create: `benchmarks/cross_branch_performance.py`

**Steps:**

1. Add tests for rewriting integer domain declarations without changing other
   model text.
2. Add tests for serializing representative core, C2, cardinality, and unary
   catalog cases into source accepted by the current parser.
3. Add tests for resource-failure truncation keys and worker JSON parsing.
4. Run the focused tests and confirm they fail before implementation.

### Task 2: Implement the branch worker and resource monitor

**Files:**
- Create: `benchmarks/cross_branch_worker.py`
- Modify: `benchmarks/cross_branch_performance.py`

**Steps:**

1. Implement typed/legacy API detection and fresh parsing for every solver
   invocation.
2. Emit exactly one JSON record containing status, result, and solver time.
3. Launch workers in their own process groups and sample aggregate RSS with
   `psutil`; terminate on timeout or memory excess.
4. Test success, solver error, malformed output, timeout, and memory status
   handling.

### Task 3: Implement branch isolation, execution, and reports

**Files:**
- Modify: `benchmarks/cross_branch_performance.py`

**Steps:**

1. Resolve immutable commit hashes and prepare/reuse a detached `modk`
   worktree with its own uv environment.
2. Collect catalog workloads and common scalable model files, with explicit
   exclusions for fixed domains and branch-only files.
3. Execute the four branch/algorithm configurations in increasing domain order
   and apply resource-failure truncation.
4. Write atomic CSV, Markdown summary, runtime plot, and memory plot outputs.

### Task 4: Verify and run the comparison

**Files:**
- Create: `benchmarks/results/cross_branch/results.csv`
- Create: `benchmarks/results/cross_branch/summary.md`
- Create: `benchmarks/results/cross_branch/runtime.png`
- Create: `benchmarks/results/cross_branch/memory.png`

**Steps:**

1. Run focused tests, compileall, and `git diff --check`.
2. Run a small real smoke matrix against both commits and inspect result
   equality and resource readings.
3. Run the selected catalog grids and scalable model grids with 30-second and
   4-GiB limits.
4. Inspect CSV status totals and mismatches, then render and visually verify
   both plots and the Markdown summary.
