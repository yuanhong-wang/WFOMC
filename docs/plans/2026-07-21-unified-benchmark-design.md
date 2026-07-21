# Unified Benchmark Design

## Goal

Provide one authoritative benchmark path for comparing current WFOMC
algorithms under either cold-process or compile-once/domain-series execution,
while keeping historical cross-branch experiments reproducible but clearly
separate.

## Architecture

`benchmarks/domain_series_performance.py` becomes the canonical current-version
runner.  It consumes one workload inventory for both protocols, supports the
catalog and model files, and writes a versioned manifest plus one result schema.
The cold protocol creates an isolated runtime for every measured solve.  The
compile-once protocol creates one compilation and runtime per repetition and
then evaluates increasing domains that share the same domain-free problem key.

The existing cross-branch runner remains available because old branches expose
a different API.  It receives the same safety fixes where possible: sources are
read from the selected commit, resume rows must match the complete run identity,
and resource truncation is scoped to one branch/algorithm/problem series.  The
boundary-profile historical runner remains a historical-report tool and says so
explicitly; it must not be treated as a current head-to-head comparison.

## Workload Identity and Validation

Every workload carries both an exact input hash and a domain-free series hash.
The latter is calculated after replacing only the integer domain declaration;
domain-dependent cardinality bounds therefore produce different series.  Model
source is read with `git show <commit>:<path>`, never from an unrelated working
tree.  Inputs whose evidence references constants outside the generated domain
are classified as `invalid`, and known algorithm capability failures are
classified as `unsupported` rather than generic errors.

Catalog metadata (`comparison_group` and `correction_divisor`) travels with the
workload.  Same-input correctness requires at least two successful algorithms.
Alternative encodings are compared after exact rational normalization by their
correction divisors.

## Persistence and Statistics

Each output directory contains `manifest.json`.  Its run id covers protocol,
workload inventory, commits, algorithms, timeout, memory limit, repetitions,
Python/platform metadata, lockfile hash, and relevant dirty-tree hash.  Resume
is permitted only when the manifest matches exactly; otherwise the runner stops
with an actionable error.

Results retain median, minimum, maximum, individual timing samples, parse time,
compile time, wall time, status, and resource information.  Summary tables keep
success/timeout/unsupported/invalid/error separate, report paired-success speed
only as such, and show solved-under-budget counts so timeouts are not silently
removed from the headline result.  Compile-once summaries distinguish first
domain setup cost, later warm-domain cost, and complete-series time.

## Compatibility

Existing result files remain readable as legacy artifacts but are never resumed
into a new run without a matching manifest.  Existing helper APIs used by tests
remain available while the canonical CLI and schema move to the unified path.

## Verification

Unit tests cover series identity, manifest mismatch rejection, commit-sourced
models, per-configuration truncation, status classification, singleton
correctness, normalized comparison groups, and preservation of timing samples.
Smoke tests run all three algorithms under both protocols and assert result
agreement plus boundary-profile template reuse.
