# Flat Benchmark Layout

## Goal

Keep `benchmarks/` limited to the current case catalog, one current runner, and
its README.  Remove public benchmark selectors and all historical runners and
result artifacts.

## Layout

```text
benchmarks/
  README.md
  cases.py
  run.py
```

`cases.py` exposes one deterministic tuple containing all 165 concrete cases
and a key lookup.  It has no smoke/main/exhaustive suites and accepts no suite
name.  Case metadata such as category and family remains available for result
reporting, but it cannot select the inventory.

`run.py` benchmarks that complete catalog with boundary-profile, Fast, and
Incremental3.  It retains cold and compile-once measurement protocols, resource
limits, resume manifests, correctness checks, and hidden worker mode.  It drops
model-source discovery, legacy-branch compatibility, `--suite`, and `--sources`.
Results default to the repository-level `benchmark-results/` directory so a run
does not make `benchmarks/` messy again.

## Removal

Delete the historical boundary-profile and cross-branch runners, their worker,
their unit tests, every tracked file below `benchmarks/results/`, Python cache
files, and Finder metadata.  Git history remains the archive for those files.

## Verification

Tests assert that the catalog has exactly 165 unique cases, the CLI has no
selector flags, manifests contain no selector fields, and all three algorithms
still agree on representative small cases.  Run focused benchmark tests and the
full repository test suite, and verify the final directory contains only the
three intended source files.
