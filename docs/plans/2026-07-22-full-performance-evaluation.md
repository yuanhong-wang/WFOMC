# Full Performance Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run all 165 benchmark cases with Fast, Incremental3, and BP-DP and present reproducible English results in the AAAI paper.

**Architecture:** Extend the flat runner's algorithm inventory from FastV2 to Fast, run the complete catalog under the cold protocol, and analyze the resulting CSV with a standalone plotting script. The paper consumes a vector figure and reports only statistics computed from the saved raw results.

**Tech Stack:** Python 3.11, WFOMC typed API, pytest, matplotlib/seaborn, CSV/JSON, LaTeX/AAAI 2027, Poppler.

---

### Task 1: Benchmark the requested algorithms

**Files:**
- Modify: `benchmarks/run.py`
- Modify: `tests/unit/test_benchmark_run.py`

1. Add a failing assertion that the public benchmark algorithms are exactly
   `boundary-profile`, `fast`, and `incremental3`.
2. Replace FastV2 with Fast in the runner inventory and English labels.
3. Run focused runner tests.
4. Run a small three-algorithm smoke comparison and verify identical results.

### Task 2: Run the complete experiment

**Output:**
- Create: `benchmark-results/paper-full-2026-07-22/manifest.json`
- Create: `benchmark-results/paper-full-2026-07-22/results.csv`
- Create: `benchmark-results/paper-full-2026-07-22/summary.md`

1. Run `benchmarks/run.py` with `--protocol cold`, three repetitions, a
   30-second timeout, 4 GiB, and the three requested algorithms.
2. Monitor until all 495 case/algorithm rows finish.
3. Verify result completeness, consensus, failure statuses, and manifest
   identity.

### Task 3: Generate publication artifacts

**Files:**
- Create: `scripts/plot_benchmark_results.py`
- Create: `/Users/lucien/Sync/overleaf/reorder_wfomc/bp-draft/figures/benchmark_performance.pdf`
- Create: `/Users/lucien/Sync/overleaf/reorder_wfomc/bp-draft/figures/benchmark_performance_summary.json`

1. Parse and validate the 495-row result CSV.
2. Compute coverage, median times, paired geometric means, and per-category
   speedups without counting failed runs as finite runtimes.
3. Render the two-panel colorblind-safe vector figure and machine-readable
   summary.
4. Inspect the figure visually at publication scale.

### Task 4: Write and verify the English paper section

**Files:**
- Modify: `/Users/lucien/Sync/overleaf/reorder_wfomc/bp-draft/sections/06_experiments.tex`
- Modify: `/Users/lucien/Sync/overleaf/reorder_wfomc/bp-draft/main.tex` only if a package or figure path is required.

1. Replace experiment placeholders with setup, metrics, results, figure,
   compact table, and limitations in English.
2. Insert only values computed from the saved summary.
3. Compile `main.tex`, render the experiment pages, and fix every overflow,
   clipping, or readability problem.
4. Run focused and full repository tests and `git diff --check`.
