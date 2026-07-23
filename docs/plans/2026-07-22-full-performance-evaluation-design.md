# Full Performance Evaluation Design

## Scope

Evaluate `fast`, `incremental3`, and `boundary-profile` (reported as BP-DP) on
all 165 concrete benchmark cases.  Use the cold protocol so every repetition
gets a fresh runtime and no cross-domain cache reuse influences the comparison.
Each case receives a 30-second solve limit, a 4-GiB process-tree RSS limit, and
three repetitions; successful runtimes are reported by their median.

## Data and failure handling

The flat benchmark runner writes one row per case and algorithm, together with
the complete manifest and environment metadata.  Timeouts, memory limits,
unsupported inputs, and solver errors remain in the results and contribute to
coverage, but runtime ratios use only paired successful cases.  Alternative
encodings are checked after applying their catalog correction divisors.

## Presentation

The main English figure has two panels.  The first is a log-scale cactus plot:
for each method it shows how many of the 165 cases finish within a given median
runtime.  The second shows paired `competitor / BP-DP` speedups on a log scale,
split by benchmark category and rendered as lightly jittered observations over
box summaries.  A horizontal line at one separates BP-DP wins from losses.

An English table reports solved cases, timeout/memory/unsupported counts,
median successful runtime, and paired geometric-mean speedup relative to BP-DP.
The experimental text states the hardware, cold protocol, limits, number of
repetitions, coverage, paired statistics, and threats to validity.  The figure
uses a colorblind-safe palette, vector PDF output, consistent method labels,
and typography sized for the AAAI two-column layout.

## Artifacts

Raw data stays in `benchmark-results/paper-full-2026-07-22/`.  A reproducible
plotting script lives in `scripts/plot_benchmark_results.py`; the paper receives
the vector figure in `figures/benchmark_performance.pdf` and the updated English
experiment section in `sections/06_experiments.tex`.
