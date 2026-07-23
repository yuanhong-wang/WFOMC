# FastWFOMC-Style Experimental Comparison

## Goal

Match the experimental exposition of van Bremen and Kuzelka (2021) while
retaining the rigor and complete coverage of the validated 165-case run.

## Alternatives

1. **Replace the aggregate figure with a four-family scaling figure
   (selected).** This most closely matches FastWFOMC: one log-runtime panel for
   each of 3-regular graphs, 4-coloured graphs, derangements, and three
   matchings. The full-catalog table and aggregate paired ratios remain in the
   text.
2. Add the four-family figure alongside the aggregate figure. This preserves
   every visualization but consumes too much main-paper space and repeats the
   same performance message.
3. Keep the aggregate figure and only rewrite the prose. This is compact but
   does not match FastWFOMC's visual comparison.

## Data and Presentation

The figure is generated from the completed cold-cache CSV and does not rerun or
alter the benchmark protocol. It uses the FO2-cardinality-reduction variants
for 3-regular graphs, derangements, and three edge-disjoint perfect matchings,
matching FastWFOMC's one-oracle-call comparison convention. The direct FO2
encoding is used for 4-coloured graphs. Each panel plots median exact solve time
against domain size on a logarithmic y-axis. Downward triangles at 30 seconds
denote timeouts; crosses denote the memory limit. All three algorithms use the
same colors and markers in every panel.

The paper follows FastWFOMC's organization: implementation and comparison
conditions, benchmark definitions and encoding notes, then one paragraph per
panel with concrete timings and an explicit negative or inconclusive result
where appropriate. A final paragraph reports full-catalog coverage and paired
geometric means, so the selected-family view does not replace the complete
evaluation.

## Verification

- Unit-test family selection and failure handling with a synthetic CSV.
- Generate the vector PDF from the validated 495-row result file.
- Compile the AAAI paper and render every changed page to PNG.
- Run the full Python test suite and both repositories' diff checks.
