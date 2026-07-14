# Benchmark Case Migration Design

Date: 2026-07-13

## Goal

Migrate every benchmark problem family and parameter grid from
`new_WFOMC_with_notes/benchmarks` into the current repository as a reusable
typed case catalog.

## Boundary

This migration contains cases only. It does not migrate Opt27/Opt22 or Wang
baselines, solver-comparison runners, profiling scripts, historical CSVs, or
old private execution APIs.

`benchmarks/cases.py` is the single authority for formula templates, weights,
domain sizes, cardinality constraints, categories, default algorithms, and
reporting correction factors. Each expanded `BenchmarkCase` builds a current
typed `Problem`; future runners can choose how to time it.

The catalog preserves:

- the 4-row smoke, 16-row main, and 49-row exhaustive transformed-FO2 suites;
- C2 3-regular, coloured 3-regular, directed 3-in/3-out, and direct-vs-hand
  case grids;
- properly 4-coloured 3-regular and directed 3-regular cardinality cases;
- unary unconstrained, exact, interval, structure, and clique-gate workloads.

The duplicate source definition of `psi_k_regular_transformed()` is not
carried over; only the effective canonicalized definition is migrated.

## Verification

Tests enforce suite cardinalities, globally unique deterministic keys, typed
problem construction, declared-predicate constraints, and representative case
metadata. The focused catalog test, complete test suite, compileall, and
`git diff --check` finish the migration. Ruff is optional because it is not a
dependency of this repository.
