# WFOMC benchmarks

The benchmark entry points are:

- `cases.py`: the complete deterministic catalog of 165 concrete problems;
- `run.py`: run all 165 cases;
- `run_paper.py`: run the 336 fixed-grid Core cases used by the paper
  experiments;
- `run_decomposition_ablation.py`: compare the BP-tree candidate strategies
  on the four eight-type sparse-interaction graphs.

The runners have no inventory selectors.  Each entry point owns one explicit
catalog, while both runners use the same BP-DP, Fast, and Incremental3
measurement implementation.

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
  --memory-gib 4
```

For the paper's planner ablation, run:

```console
uv run python -m benchmarks.run_decomposition_ablation \
  --domain-size 40 \
  --timeout 300 \
  --memory-gib 4 \
  --repetitions 1 \
  --out benchmark-results/decomposition-ablation
```

The ablation records the selected strategy, BP- and join BP-widths, planner
work estimates, peak RSS, exact-result hash, and cold runtime for
heuristic-only, exact subset-cost, forced tail-caterpillar, and forced greedy
agglomeration.

The paper runner fixes all 28 problem-specific domain grids in
`run_paper.py`.  It defaults to three cold repetitions, a 300-second timeout,
and early stopping for larger domains in the same algorithm/family series.
Four typed sparse-interaction families, five properly coloured regular-graph
configurations, Friends & Smokers, Academic Advising, ID2 Gene Regulation,
the IMDB WorkedUnder FO2 fragment, and WebKB link classification are
paper-only and do not extend the general catalog in `cases.py`.

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
core/typed-path-relation-k8/n120
core/properly-4-coloured-undirected-3-regular/fo2-cardinality-reduction/n20
core/properly-4-coloured-undirected-4-regular/fo2-cardinality-reduction/n20
core/friends-smokers/n100
core/academic-advising/n20
core/id2-gene-regulation/n14
core/imdb-worked-under-fo2/n64
core/webkb-link-classification/n20
c2/undirected-3-regular/direct-c2/n20
c2/undirected-3-regular/fo2-cardinality-reduction/n20
unary/bi-total-relation/sx-cardinality/exact/n100
```

For cases 1--4 and 9--14 of the original 14-family Core suite, Incremental3
receives the original C2
sentence with counting quantifiers.  BP-DP and Fast receive the corresponding
weighted-Skolem FO2 matrix completed by binary cardinality constraints.  The
CSV records this algorithm-specific choice in `input_variant`.  An undirected
`k`-regular reduction carries a `(k!)^n` correction divisor, while its original
C2 input has divisor one.  The five properly coloured regular-graph
configurations use the same split: BP-DP and Fast receive the reduction, while
Incremental3 receives the parameter-matched original C2 sentence.
