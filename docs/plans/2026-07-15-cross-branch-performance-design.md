# Cross-Branch Performance Benchmark Design

Date: 2026-07-15

## Goal

Compare `fastv2` and `incremental3` on the current `devel` branch and the
historical `modk` branch as domain size grows.  Cover both the reusable
`benchmarks/cases.py` catalog and scalable repository model files, while
enforcing a 30-second wall-clock limit and a 4 GiB process-tree RSS limit per
measurement.

## Architecture

`benchmarks/cross_branch_performance.py` is the controller and report writer.
It records the exact commit for each branch, prepares an isolated `modk` git
worktree, converts catalog cases and model files into portable `.wfomcs` text,
and invokes `benchmarks/cross_branch_worker.py` in a fresh process for every
configuration.  The worker contains a small API adapter for the typed `devel`
API and the legacy `modk` API, but both branches parse the same source text.
Module import and parsing happen before the solve timer; parse time is recorded
separately so algorithm curves show only `solve()` / `wfomc()` execution.

The controller monitors the complete child process tree with `psutil`.  A run
is terminated and recorded as `timeout` after 30 seconds, or as `memory` after
the aggregate resident set exceeds 4 GiB.  Peak observed RSS, wall time,
solver-reported result, stderr, branch, algorithm, case family, and domain size
are retained in CSV.  Unsupported inputs and parse/solver failures remain in
the report instead of silently disappearing.

## Workloads and stopping rule

Benchmark catalog cases retain the domain grids in `benchmarks/cases.py`.
Repository `.wfomcs` and `.mln` files whose domain declaration is an integer
are expanded over `2, 4, 8, 16, 32, 64`; fixed named-domain files are excluded
with an explicit reason.  Only files present on both commits are used for the
cross-branch model comparison.

For each case family, larger domain sizes are skipped for all four
branch/algorithm configurations after any configuration first returns
`timeout` or `memory`.  The other configurations at that same domain still run
so the boundary point is complete.  Ordinary solver errors do not trigger
resource truncation because support can vary with a particular model.
Runs default to one measured repetition to keep the exhaustive suite bounded;
the CLI supports additional repetitions and reports their median.

## Outputs and verification

Results are written below `benchmarks/results/cross_branch/`.  `results.csv`
is atomically checkpointed after every domain case, while the human-facing
reports are rendered after the complete run:

- `results.csv`: complete machine-readable rows;
- `summary.md`: commits, limits, status totals, correctness mismatches, and
  compact family/configuration comparisons;
- `runtime.png`: runtime curves versus domain size;
- `memory.png`: peak-RSS curves versus domain size.

Tests cover portable serialization, integer-domain rewriting, adaptive
truncation, worker result parsing, and report generation.  A smoke run exercises
both real branch environments before the full benchmark is launched.
