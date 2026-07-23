# WFOMC benchmarks

The benchmark entry points are:

- `cases.py`: the complete deterministic catalog of 165 concrete problems;
- `run.py`: run all 165 cases;
- `run_paper.py`: run the 53 Core cases used by the paper experiments.

The runners have no inventory selectors.  Each entry point owns one explicit
catalog, while both use the same BP-DP, Fast, and Incremental3 measurement
implementation.

## Run

```console
uv run python benchmarks/run.py \
  --protocol compile-once \
  --repetitions 3 \
  --timeout 30 \
  --memory-gib 4
```

For the paper experiment, run:

```console
uv run python benchmarks/run_paper.py \
  --protocol cold \
  --repetitions 3 \
  --timeout 300 \
  --memory-gib 4
```

The full runner defaults to `benchmark-results/`; the paper runner defaults to
`benchmark-results/paper-core/`.  Use `--out` to choose a different location.
A run writes `manifest.json`, `results.csv`, and `summary.md`; resumption is
allowed only when the complete manifest identity matches.  Use `--no-resume`
to replace an existing run.

`cold` gives every measured solve a fresh runtime.  `compile-once` groups only
cases with exactly the same domain-free problem, compiles once per repetition,
and evaluates the group's domains in increasing order.  These protocols answer
different questions and should not be combined into one speed ratio.

Statuses distinguish unsupported inputs, invalid inputs, timeouts, memory
limits, and worker failures.  Correctness compares algorithms on identical
cases and applies each case's correction divisor when comparing alternative
encodings of the same mathematical problem.

## Case keys

Keys describe the mathematical problem and encoding.  Representative examples
are:

```text
core/permutations/fo2-cardinality-reduction/n100
core/derangements/fo2-cardinality-reduction/n100
core/undirected-3-regular/fo2-cardinality-reduction/n60
core/3-edge-disjoint-perfect-matchings/fo2-cardinality-reduction/n20
c2/undirected-3-regular/direct-c2/n20
c2/undirected-3-regular/fo2-cardinality-reduction/n20
unary/bi-total-relation/sx-cardinality/exact/n100
```

Core permutations, derangements, endofunctions, regular graphs, and perfect
matchings use weighted-Skolem FO2 matrices completed by binary cardinality
constraints.  An undirected `k`-regular reduction carries a `(k!)^n` correction
divisor.
