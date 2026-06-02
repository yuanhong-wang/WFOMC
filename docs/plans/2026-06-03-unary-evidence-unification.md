# Unary Evidence Unification Plan

## Goal

Represent unary evidence once, expose only a small public strategy API, and let
each WFOMC algorithm consume the same evidence-profile partition in the way that best
matches its own computation.

## Public API

Keep one selector:

```python
class UnaryEvidenceStrategy(Enum):
    AUTO = "auto"
    CCS = "ccs"
```

- `AUTO` is the default.
- `CCS` forces the modular auxiliary-predicate and cardinality-constraint
  encoding.
- Remove `UnaryEvidenceEncoding` and internal public strategy values such as
  `CONFIG`, `THREADED`, and `EXPANDED`.

`AUTO` resolves as follows:

| Algorithm | Effective path |
| --- | --- |
| Standard | evidence-profile configuration coefficients |
| Fast | CCS fallback |
| Fastv2 | evidence-expanded cell graph view |
| Incremental | threaded evidence-profile capacities |
| Incremental3 | evidence-profile configuration coefficients |
| Recursive | CCS fallback |

## File Layout

```text
src/wfomc/
  context/
    unary_evidence.py
    unary_cardinality.py
    wfomc_context.py
    incremental3_context.py
  cell_graph/
    cell_graph.py
    components.py
    utils.py
```

`context/unary_evidence.py` owns:

- `UnaryEvidenceStrategy`
- `EvidenceProfile`
- `UnaryEvidencePartition`
- `CellEvidenceAllocation`
- `CellConfigCoefficientBasis`
- `UnaryEvidencePlan`
- the private CCS encoder

The separate `unary_evidence/`, `legacy_ccs.py`, and `evidence_graph.py` files
are unnecessary. The Fastv2 evidence graph is a cell graph implementation and
belongs in `cell_graph.py`.

## Responsibilities

### UnaryEvidencePartition

Own only evidence semantics:

- group named ground unary evidence into deterministic fixed-size evidence profiles
- expose the predicates required for cell construction
- build a coverage formula
- compile against concrete cells

### CellEvidenceAllocation

Own only cell-graph combinatorics:

- cell-to-evidence-profile compatibility
- evidence-profile capacities
- configuration coefficients
- threaded capacity transitions
- exact normalization

### UnaryEvidencePlan

Own strategy-specific preparation:

- validate the exchangeability requirement
- build the semantic unary-evidence partition
- choose coverage formula or CCS formula
- expose cardinality constraints and repeat factor
- compile cell allocations only for `AUTO`

### WFOMCContext

Apply prepared results without duplicating strategy logic:

- conjoin the prepared formula
- extend cardinality constraints when present
- apply the repeat factor
- pass required predicates and evidence metadata into cell graph construction
- expose `cell_evidence_allocation(cells)`

### Algorithms

Algorithms do not inspect internal unary evidence strategy names:

- Standard and Incremental3 call `iter_config_coefficients`
- Incremental threads `next_remaining_counts` transitions
- Fastv2 uses the evidence-expanded optimized cell graph
- Fast and Recursive receive CCS-prepared contexts under `AUTO`

## Verification

1. Unit-test evidence-profile grouping, compatibility, configuration coefficients,
   and threaded normalization.
2. Test `AUTO` and explicit `CCS` across all supported algorithms.
3. Test Incremental3 with linear order and unary evidence through the
   evidence-profile configuration coefficient path.
4. Verify evidence-only predicates are assigned and weighted exactly once.
5. Run the full test suite and `git diff --check`.
