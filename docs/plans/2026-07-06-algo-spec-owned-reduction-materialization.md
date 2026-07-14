# Algo-Spec-Owned Reduction and Materialization

Status: completed as framework direction
Date: 2026-07-06

## Goal

Make `engine` pure orchestration and make every algorithm enter the framework
through `algo.spec`.

## Final Shape

Each algorithm declares:

```python
SPEC = AlgoSpec(
    name=...,
    resolve_options=...,
    reduce=...,
    materialize=...,
    run=...,
)
```

Engine flow:

```text
analyze_problem
  -> spec.resolve_options
  -> cache/get spec.reduce
  -> cache/get spec.materialize
  -> spec.run
```

## Rules Locked In

- `engine` does not branch on algorithm names.
- `reduction` does not import `AlgoName`.
- `reduction` exposes target functions such as `reduce_to_ufo2` and
  `reduce_to_counting_dp`.
- `ReducedProblem` has `reduction_kind`, not `algo` or `context_kind`.
- Algorithm-owned inputs stay under `algo/<name>/`.

## Remaining Related Work

This plan does not remove the typed-FOL-to-cell-graph bridge. That belongs to
the FOL formula migration plan.
