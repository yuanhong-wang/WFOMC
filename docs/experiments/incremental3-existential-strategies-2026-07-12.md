# Incremental3 existential strategy experiment (2026-07-12)

## Setup

- Apple Silicon/macOS workspace, Python 3.11 through `uv`.
- Exact arithmetic and identical model input for each pair.
- Five warmups and 50 repetitions for `existential.wfomcs`; three warmups and
  30 repetitions for the other models.
- Every benchmark command was wrapped in a 30-second hard timeout.
- Times below are median end-to-end `parse_problem_file + solve` times in one Python
  process, so interpreter startup is excluded.

## Results

| Model | Domain | Counting | Skolem | Faster strategy | Exact results equal |
|---|---:|---:|---:|---|---|
| `existential` | 7 | 47.287 ms | 34.065 ms | Skolem (28%) | yes |
| `friends-smokes` | 10 | 45.115 ms | 39.475 ms | Skolem (13%) | yes |
| `function-no-fix` | 5 | 25.015 ms | 24.701 ms | effectively tied | yes |
| `nonisolated_graph` | 10 | 27.162 ms | 26.075 ms | Skolem (4%) | yes |
| `permutation-no-fix` | 5 | 35.961 ms | 40.820 ms | Counting (12%) | yes |

For `existential.wfomcs`, both strategies returned
`15515568475732467854453889`. A separate one-shot process measurement was
0.24 seconds for counting and 0.23 seconds for Skolem, showing that startup
dominates short CLI invocations.

## Conclusion

Neither reduction dominates. Weighted Skolemization is usually a little
faster on these small models and substantially faster on the case with two
independent relations; native counting wins on `permutation-no-fix`. Counting
remains the incremental3 default because it expresses the source constraint
directly, avoids negative auxiliary weights, and handles composite bodies via
an explicit definition predicate. Users can select Skolem when it benchmarks
better for their workload.
