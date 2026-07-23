# Paper Experiment Rerun Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-run the complete paper benchmark after the automatic exact and
truncated-series arithmetic changes, regenerate publication artifacts, and
replace every stale experimental claim with values derived from the new CSV.

**Architecture:** Preserve the paper's cold-cache protocol and complete
165-case catalog so the new measurements are directly comparable with the
existing evaluation. Store the new manifest and raw CSV in a fresh result
directory, generate all figures and summary JSON from that CSV, then update
and compile the AAAI LaTeX draft without touching unrelated paper edits.

**Tech Stack:** Python 3.11, python-flint, pytest, psutil, matplotlib, CSV/JSON,
AAAI 2027 LaTeX, latexmk, Poppler.

---

### Task 1: Freeze and validate the experiment protocol

**Files:**

- Read: `benchmarks/README.md`
- Read: `benchmarks/run.py`
- Test: `tests/unit/test_benchmark_run.py`
- Test: `tests/unit/test_plot_benchmark_results.py`

**Steps:**

1. Confirm the catalog contains 165 cases and the algorithms are BP-DP, Fast,
   and Incremental3.
2. Confirm the paper describes the runner's cold protocol: fresh runtime per
   repetition, three repetitions, 30-second per-solve timeout, and 4 GiB RSS.
3. Run benchmark-runner and plotting tests before starting the long run.
4. Record the current source commit, dirty-tree digest, Python version, OS,
   processor, and memory in the generated manifest.

### Task 2: Run the complete cold benchmark

**Files:**

- Create: `benchmark-results/paper-full-auto-series-300s-2026-07-22/manifest.json`
- Create: `benchmark-results/paper-full-auto-series-300s-2026-07-22/results.csv`
- Create: `benchmark-results/paper-full-auto-series-300s-2026-07-22/summary.md`

**Steps:**

1. Run `uv run python benchmarks/run.py --protocol cold --repetitions 3
   --timeout 300 --memory-gib 4 --order-seed 0 --no-resume` with the fresh
   output directory.
2. Monitor the worker until all 495 method--case rows are recorded.
3. Verify the manifest matches the intended protocol and current dirty-tree
   digest.
4. Verify there are exactly 165 cases and 495 rows, no worker errors, no
   answer mismatches, and no alternative-encoding mismatches.

### Task 3: Regenerate paper figures and computed statistics

**Files:**

- Read: `scripts/plot_benchmark_results.py`
- Replace: `/Users/lucien/Sync/overleaf/reorder_wfomc/bp-draft/figures/benchmark_performance.pdf`
- Replace: `/Users/lucien/Sync/overleaf/reorder_wfomc/bp-draft/figures/benchmark_family_scaling.pdf`
- Replace: `/Users/lucien/Sync/overleaf/reorder_wfomc/bp-draft/figures/benchmark_performance_summary.json`

**Steps:**

1. Generate both vector figures and the machine-readable summary from the new
   result CSV.
2. Cross-check every aggregate in the summary against an independent CSV
   query.
3. Render both PDFs at publication scale and inspect labels, legends, markers,
   clipping, and readability.

### Task 4: Update and verify the AAAI paper

**Files:**

- Modify: `/Users/lucien/Sync/overleaf/reorder_wfomc/bp-draft/sections/06_experiments.tex`
- Modify if needed: `/Users/lucien/Sync/overleaf/reorder_wfomc/bp-draft/main.tex`
- Replace: `/Users/lucien/Sync/overleaf/reorder_wfomc/bp-draft/main.pdf`

**Steps:**

1. Update coverage, failure counts, paired ratios, family timings, speedups,
   correctness statements, and hardware metadata using only the saved results.
2. Preserve the existing FastWFOMC-style exposition and state negative or
   neutral findings precisely.
3. Compile `main.tex` with latexmk and fail on unresolved references or LaTeX
   errors.
4. Render and inspect the experiment pages for overflow, clipping, figure
   legibility, and table placement.
5. Run plotting tests, the full repository test suite, both repositories'
   `git diff --check`, and report all changed artifacts.
