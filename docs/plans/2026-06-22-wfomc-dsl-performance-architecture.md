# WFOMC DSL Performance Architecture Implementation Plan

> **Last updated:** 2026-06-23 (re-scoped after the `wfomc.framework` front-end landed and after a
> review of the reference implementation in `/Users/lucien/Sync/repos/new_WFOMC_with_notes`).

**Goal:** Turn the `wfomc.framework` pipeline into a production-grade, high-performance WFOMC solver
with a complete and benchmarked path from source text to an exact count, while preserving the legacy
solver as a correctness oracle during migration.

**Architecture:** A staged compiler pipeline: source text → parsed AST → typed immutable Core AST →
normalization passes → `NormalForm` → **CellGraph IR (domain-size-independent)** → solver plan →
algorithm execution. Each stage owns its caches and exposes stable, benchmarked data structures.

**Tech Stack:** Python 3.11, Lark, an internal Boolean WMC backend (CNF/Tseitin + cached DPLL,
optional PySAT shortcut), python-flint ring elements / truncated polynomials, pytest,
pyinstrument or cProfile.

---

## Current Status (2026-06-23)

### Built — `wfomc.framework` front-end (parse → normalize → ground)
- `syntax/`: immutable `Core*` AST (`frozen`, `slots`), builders, traversal (`iter_children`,
  `pre_order`, `post_order`), and a legacy `lower.py` bridge from the SymPy-backed `QFFormula`.
- `normal_form/`: `NormalForm` (universal / witness / counted-definition clauses) plus the
  `normalize()` pipeline (alpha-renaming → fragment validation → count abstraction →
  implication elimination → NNF → prenex lifting → clause collection).
- `problem/`: parsed `Problem` model with domain/weight/evidence validation.
- `parser/`: Lark grammar, cached parser factory (`get_lark_parser` via `lru_cache`), per-call
  transformer that isolates `name2pred`.
- `grounding/`: `BitsetProblem` with integer-indexed predicates/atoms and a `BoolDag`
  (hash-consed Boolean DAG) for grounding quantifier-free formulas.
- `boolean/`: `BoolDag`, `BoolAssumptions`; `simplify.py` is an empty placeholder.
- `backend/`: `Backend` protocol + `BackendResult` — **contract only, no implementation**.
- `compiler/`, `runtime/`, `cli.py`, `benchmark.py`: parse-to-grounded composition, options/diagnostics,
  and a parse/ground timing harness. 72 framework tests pass.

### Missing / blocking production
1. **No solving backend.** `compile_problem_text()` stops at `BitsetProblem`; nothing implements
   `Backend.solve()`. The CLI and benchmarks measure parse + ground only — never an actual count.
2. **Framework is an island.** Nothing in `framework/` imports `algo/`/`cell_graph/`, and nothing
   legacy imports `framework/`. There is no end-to-end path through the new pipeline.
3. **Full ground-atom enumeration contradicts lifted inference.** `_build_ground_atom_index`
   materializes every atom (`O(P·n²)` for binary predicates) and evidence is a `1<<atom_id` bignum.
   Lifted WFOMC must work over 1-/2-types (cells), whose structure is independent of domain size `n`.
   The bitset grounding is acceptable only as a small-instance / unary-evidence fallback.
4. **Correctness landmine:** cardinality coefficients/values are `float` (`problem/model.py`,
   `grounding/bitset.py`) — must be exact (`int`/`Fraction`/`Rational`).
5. **Maintainability debt:** `_free_vars`, `_contains_quantifier`, predicate-collection, and
   `_quantify_forall` are reimplemented in 3–4 modules; private helpers are imported across packages;
   several 7-line "boundary" modules re-export `bitset.py`/`normalize.py` without owning real code.
6. **Tooling gap:** no `ruff`/`mypy` config or CI gate despite the project rule mandating them;
   `dataclasses` backport and heavy unused deps (`pandas`, `sympy`, `symengine`, `cmd2`,
   `prettyprinttree`, `pynauty`) are non-optional.

---

## Borrowed Ideas From the Reference Implementation

These are reusable *ideas and structures* from `new_WFOMC_with_notes` (excluding its tail-signature DP
algorithm). Borrow the design; keep our cleaner **integer-indexed** representation instead of the
reference's string-keyed atoms (`f"{p}(x)"`), which are a hot-path weakness.

| # | Idea (reference symbol) | Where it lands here | Why |
|---|--------------------------|---------------------|-----|
| B1 | **Precompile-once / instantiate-many** split (`PrecompiledCellGraph`, `precompile_cells_and_relations_from_psi` + `instantiate_cells_and_relations_from_precompiled`) | `ir/cell_graph.py`, `planning/cell_graph_builder.py` | Cell-graph *structure* depends only on ψ(x,y); weights & domain size are plugged in later. Enables parse-once/solve-many and the "reusable plans" NFR. **Highest-value borrow.** |
| B2 | **CNF/Tseitin + cached DPLL weighted model count, with a small-model PySAT shortcut** (`tseitin_cnf`, `sat_models_weighted_sum`, `_PYSAT_SMALL_MODEL_ENUMERATION_LIMIT=128`) | the Boolean WMC backend (`boolean/wmc.py`) | This *is* ADR-0005's "Boolean WMC backend," already mature. Replaces SymPy model enumeration for cell/two-table weights. |
| B3 | **Free-predicate / neutralize factoring** (`neutralize_atom_weights`, `free_one_type_unary`, `free_offdiag_factor`) | cell-graph builder | Predicates absent from ψ contribute a closed-form `(wt+wf)` factor instead of being enumerated — keeps the cell graph small and fixes the over-grounding problem (Missing #3). |
| B4 | **Flint-backed `TruncatedPolynomial` with lazy/cached coefficients** (`compare=False, hash=False` derived fields) | `ir/weights.py` cardinality path | Exact truncated power series for cardinality constraints; lazy coeff materialization + hash caching. We already depend on python-flint. |
| B5 | **Batched WMC over a vector of weight maps** (`sat_models_weighted_sum_batch`, `instantiate_..._batch`) | cardinality / parameter sweeps | One structural DPLL traversal, many weight maps — shares unit-prop/branching across a cardinality sweep. |
| B6 | **Hard native-dependency guard** (`_ensure_efficient_libraries`, `test_dependency_guards.py`) | `runtime/` + a guard test | Refuse to silently run the slow pure-Python path; raise a clear, overridable error. Prevents accidental 1000× regressions in a "performance" repo. |
| B7 | **Simplify-under-partial-assignment** (`simplify_formula_ast(ast, atom_values)`, `map_formula`) | fills `boolean/simplify.py` | Constant-fold ψ under fixed 1-type atoms before CNF, shrinking each SAT instance. Also gives us ψ(y,x)/ψ(x,x) via variable remap. |
| B8 | **Differential testing against an oracle** (`compare_with_wang_fastv2.py`, `compare_*` suites) | `tests/framework/` + benchmarks | Cross-check the new backend against the legacy adapter (and optionally the reference) on a shared case set — the safety net for every refactor. |
| B9 | **C2→FO2 reduction with overcount correction** (`c2_to_fo2.py`, `factorial_repeat_base`) | `normal_form` / backend lowering of `CountedDefinition` | Conceptual reference for lowering counting quantifiers to cardinality-constrained FO2 with a repeat/overcount factor. Borrow the *idea*, not the string-coupled code. |

---

## Non-Functional Requirements

### Correctness
- Legacy `tests/wfomc_test.py` must keep passing.
- The new backend must agree with the legacy solver on a shared case set (B8 differential tests).
- Every rewrite/lowering pass keeps golden tests for representative formulas.
- Exact arithmetic only on the count path (no `float` weights, coefficients, or thresholds).

### Performance
- No phase may regress end-to-end runtime > 5% on the benchmark suite without an ADR.
- Parser construction is amortized across repeated parses (done via `get_lark_parser`).
- CellGraph construction is domain-size-independent; instantiation is `O(cells²)` in ring ops, not in
  ground atoms.
- Solver hot loops operate on ints/tuples/arrays/ring elements — never on `Cell`/atom-string hashing.
- Native libs (python-flint, optionally PySAT) are required for the fast path (B6).

### Maintainability
- Public API compatibility for `parse_problem`, `compile_problem_text`, and a new top-level `wfomc()`.
- One implementation of each tree walk (`free_vars`, `predicates`, `contains_quantifier`) on top of
  `syntax/traversal.py`.
- Modules trend below 300 lines or are explicitly justified; "boundary" modules own real code.
- `ruff` + `mypy` gate CI.

### Observability
- Per-stage timing: parse, normalize, cell-graph build (precompile), instantiate, solve, decode.
- Benchmarks record cold and warm timings and an end-to-end **count**, not just parse/ground.

---

## Target Module Layout

```text
src/wfomc/framework/
  syntax/            # DONE: Core AST, builders, traversal, legacy lower bridge
  normal_form/       # DONE: NormalForm + normalize passes  (+ counting lowering, Task H)
  problem/           # DONE: parsed Problem model (+ exact-rational cardinality, Task A)
  parser/            # DONE: grammar + cached Lark + transformers
  grounding/         # bitset grounding -> DEMOTE to small-instance/evidence fallback (Task E)
  boolean/
    dag.py           # DONE: hash-consed Boolean DAG
    assumptions.py   # DONE
    simplify.py      # TODO: simplify-under-assignment (B7)
    wmc.py           # TODO: CNF/Tseitin + cached DPLL WMC (+ PySAT shortcut) (B2, Task D)
  ir/
    cell_graph.py    # TODO: integer-indexed CellGraph IR (precompile/instantiate) (B1, Task E)
    weights.py       # TODO: ring weights + truncated flint polynomials (B4, Task G)
  planning/
    cell_graph_builder.py   # TODO: build cells/relations from NormalForm (B1, B3)
    fast_plan.py            # TODO: solver plan dataclasses
  backend/
    interface.py     # DONE: Backend protocol + BackendResult
    adapter.py       # TODO: legacy fast_wfomc adapter backend (oracle) (Task B)
    lifted.py        # TODO: native backend over CellGraph IR (Task F)
  compiler/          # parse -> compiled problem -> solve composition
  runtime/           # options, diagnostics, native-dependency guard (B6)
  cli.py, benchmark.py
```

---

## Tasks

Ordered for adapter-first migration: get end-to-end correctness, then build the fast native path
behind the same `Backend` protocol, validated by differential tests at every step.

### Task A — Quick-win cleanups (review fallout)
**Files:** `syntax/traversal.py` (+ new `syntax/analysis.py`), `normal_form/normalize.py`,
`normal_form/model.py`, `problem/model.py`, `grounding/bitset.py`
- Add `free_vars`, `predicates`, `contains_quantifier` once on top of `iter_children`; delete the 3–4
  duplicate copies.
- Change cardinality `coefficients`/`value` from `float` to exact `int`/`Fraction`/`Rational`
  everywhere (`Problem`, `BitsetProblem`, parser).
- Replace `assert` validation in `parser/formula.py:count_parameter` with an explicit raise.
- Make cross-package helpers public (drop the underscore on names imported by `parser/problem.py`).

**Acceptance:** identical normal forms on all golden tests; no `float` on the count path; one walk impl.

### Task B — Legacy adapter backend (end-to-end + oracle)
**Files:** `backend/adapter.py`, `compiler/pipeline.py`, `cli.py`, `benchmark.py`
- Implement `Backend.solve(BitsetProblem) -> BackendResult` by lowering `NormalForm`/`Problem` into the
  existing `WFOMCContext` + `fast_wfomc` (or `incremental_wfomc`) and returning the ring count.
- Wire `compile_problem_text` → solve so `wfomc_next` prints an actual count; add `solve_s` to the
  benchmark.

**Acceptance:** `wfomc_next -i models/*.wfomcs` produces counts equal to the legacy `wfomc` CLI.

### Task C — Differential test harness (B8)
**Files:** `tests/framework/test_counts_vs_legacy.py`, `benchmarks/run_framework_benchmarks.py`
- Build a shared case registry (existing `models/*.wfomcs` plus the synthetic reference-inspired cases
  already in `benchmark.py`) and a `solve_all(case) -> {engine: RingElement}` helper.
- Mirror the reference's `compare_*` pattern (`compare_with_wang_fastv2.py`): run each case through legacy
  `wfomc` and the framework adapter; assert **exact** ring equality (no float tolerance) and, on
  divergence, report the offending case together with its parsed `NormalForm`.
- Add an opt-in `--compare-reference` cross-check against `new_WFOMC_with_notes` on the overlapping
  FO²/C² cases, auto-skipped when the reference import is unavailable.
- This harness is the regression net every later task must keep green.

**Acceptance:** counts agree across all models; the harness names the offending case and formula on
divergence; reference cross-check passes where inputs overlap.

### Task D — Boolean WMC backend (B2, B7)
**Files:** `boolean/wmc.py`, `boolean/simplify.py`, `tests/framework/test_wmc.py`
Adapt the reference's CNF/DPLL machinery to our **integer atom ids** (from `BitsetProblem`) — never the
reference's `f"{p}(x)"` strings.
- **Simplify-under-assignment (B7) → `boolean/simplify.py`:** port `simplify_formula_ast(atom_values)`
  onto the `Core*` AST / `BoolDag`: constant-fold `Top`/`Bottom` and prune `And`/`Or` children under a
  partial `{atom_id: bool}` map. Add a `map_formula`-style variable remap so callers can build ψ(y,x)
  and ψ(x,x) by substitution.
- **Tseitin CNF:** `tseitin_cnf(node, var_for_atom) -> (clauses, n_vars)` keyed by integer atom id, with
  freshly numbered aux variables; return canonicalized clauses (sorted literal tuples) for stable cache
  keys (port `_canon_clause` / `_canonicalize_cnf`).
- **Cached DPLL WMC:** port `unit_propagate` (carry the weight multiplier), branch on the shortest clause
  (`min(cnf, key=len)`), recurse via `_simplify_cnf(cnf, v, val)`, multiply free (unassigned) variables by
  `(wt + wf)`, and memoize on `(cnf_t, frozenset(assigned))` with `lru_cache`.
- **PySAT small-model shortcut (optional, flagged):** when projected models ≤ a cap (reference uses 128),
  enumerate with PySAT (`_pysat_enumerate_projected_assignments`); otherwise use the DPLL recursion.
- **API per ADR-0005:** `satisfiable(node, assumptions)`, `wmc(node, weights, assumptions) -> RingElement`,
  `projected_wmc(node, projection_atom_ids, weights, assumptions) -> dict[code, RingElement]` (the
  projection enumerates 1-type codes the cell builder consumes in Task E).

**Acceptance:** `wmc`/`projected_wmc` match a SymPy oracle on small formulas; cache keys are stable across
runs; SymPy leaves the hot path; the pure-DPLL path works with PySAT absent.

### Task E — Integer-indexed CellGraph IR (B1, B3)
**Files:** `ir/cell_graph.py`, `planning/cell_graph_builder.py`, `tests/framework/test_cell_graph_ir.py`
Port the reference's precompile/instantiate split (`PrecompiledCellGraph`) but keep everything
integer-indexed and **weight-free until instantiation**.
- **Three formula views (B7):** from the universal clause ψ(x,y) derive ψ(y,x) via the x↔y remap and
  ψ(x,x) via y→x, each `simplify`-folded; Tseitin them once (`clauses_cc` for the diagonal, `clauses_ab`
  for the off-diagonal pair).
- **Enumerate 1-types:** call `projected_wmc` (Task D) projecting onto the 1-type atoms (unary `P(x)` and
  reflexive `B(x,x)`); each surviving projection is a cell with an integer `code` bitmask.
- **Precompiled structure (domain-size-independent):** store cells with their per-cell clause sets and a
  dense pair-clause matrix `pair_clauses[i][j]` (ψ(x,y)∧ψ(y,x) with both cells' 1-type atoms fixed) —
  no weights yet, so it is reusable across weightings and domain sizes.
- **Instantiate (B3 factoring):** plug weights in via `build_weights_for_vars` + `neutralize_atom_weights`;
  compute `w_i = wmc(clauses_cc_i)`, `s_i = wmc(clauses_ab_same_i)`, and `r[i][j] = wmc(pair_clauses[i][j])`.
  Predicates absent from ψ contribute closed-form factors (`free_one_type_unary`,
  `free_one_type_reflexive`, `free_offdiag_factor = ∏(wt+wf)`) instead of being enumerated — this is what
  keeps the graph small and replaces full grounding.
- **IR shape:** `Cell = (cell_id, code, w, s)`, `relations = tuple[tuple[RingElement, ...], ...]`.
- Demote `grounding/bitset.py` to the documented small-instance / unary-evidence fallback.

**Acceptance:** precompile output is independent of weights and domain size; instantiated `w`,`s`,`r`
match direct WMC; absent predicates never appear as enumerated atoms; the fallback grounding is reachable
only for evidence/small cases.

### Task F — Native lifted backend (B1)
**Files:** `backend/lifted.py`, `planning/fast_plan.py`
- Implement `Backend.solve` over the CellGraph IR using the existing fast/incremental counting recurrence
  with integer-indexed cells and the relation matrix; no `Cell`/atom-string hashing in loops.

**Acceptance:** native backend equals the adapter on the differential suite; meets the perf NFR vs the
parse+ground baseline plus a recorded solve baseline.

### Task G — Cardinality via truncated flint polynomials (B4, B5)
**Files:** `ir/weights.py`, `backend/lifted.py`, `tests/framework/test_cardinality.py`
- **TruncatedPolynomial (B4):** port a flint-backed truncated power series with `degree_limit`, lazy
  coefficient materialization, and derived caches marked `compare=False, hash=False`; expose
  `zero/one/constant/monomial/coefficient/truncate`, and manage a backend series-length cap like
  `_ensure_backend_series_cap`.
- **Constraint as indeterminate:** for each cardinality predicate, replace its weight with a degree-1
  series (one indeterminate per constraint), carry polynomials through the cell/relation WMC, then read
  off the coefficient at the target count and apply the comparator.
- **Batched WMC (B5):** when sweeping several weight maps (e.g. constraint indeterminates), run one
  structural DPLL traversal over a vector of weight maps (`sat_models_weighted_sum_batch` /
  `instantiate_..._batch`) so unit-propagation and branching are shared.

**Acceptance:** cardinality counts match the legacy solver; the batched/poly path beats per-weight
re-solve on a multi-constraint case.

### Task H — Counting / counted-definition lowering (B9)
**Files:** `normal_form/normalize.py` or `backend/lifted.py`, `tests/framework/test_counting.py`
- Borrow the C2→FO2 reduction *idea* (`c2_to_fo2.py`): lower `CountedDefinition` / witness clauses into
  cardinality-constrained FO² over fresh witness predicates, and divide out the overcount with a
  `factorial_repeat_base`-style repeat factor — adapted to our `NormalForm` (do **not** import the
  reference's string-coupled code).
- Replace the fragile exception-driven fallback in `normalize()` (catching its own `NormalizationError`
  to switch strategies) with an explicit structural decision; if kept, chain the original via `__cause__`.

**Acceptance:** C² examples (`2-regular-graph-sc2.wfomcs`, function/derangement cases) match legacy counts;
no self-caught `NormalizationError` masks a genuine error.

### Task I — Native-dependency guard (B6)
**Files:** `runtime/dependencies.py`, `tests/framework/test_dependency_guards.py`
- Port `_ensure_efficient_libraries(context)`: detect missing python-flint / PySAT (and gmpy2 if used) and
  **raise** an actionable `RuntimeError` naming the missing libs before entering the fast path.
- Provide an explicit `_ALLOW_INEFFICIENT_FALLBACK`-style override that downgrades the error to a one-time
  `warnings.warn` + stderr notice (port the one-shot `_INEFFICIENT_LIBRARY_WARNING_EMITTED` latch).
- Add the guard test (mock the natives absent; assert it raises without the override and warns exactly
  once with it).

**Acceptance:** guard raises with an actionable message when natives are absent; override path warns once;
the fast backends call the guard at entry.

### Task J — Production polish: API, errors, boundaries, tooling
**Files:** `framework/__init__.py`, `errors.py`, the boundary modules, `pyproject.toml`,
`.github/workflows/python-app.yml`
- Narrow the public surface to the real API; keep AST/DAG internals importable but not headline.
- Route normalization/validation errors through `CompilationError`/`FrameworkError`.
- Either split `bitset.py`/`normalize.py` into the existing boundary modules or delete the placeholders.
- Add `[tool.ruff]`/`[tool.mypy]`; gate them in CI. Remove the `dataclasses` backport; move heavy deps
  to optional groups.

**Acceptance:** `ruff`/`mypy` clean in CI; slim install works without pandas/sympy; one obvious public API.

### Task K — Perf gate and docs
**Files:** `benchmarks/`, `docs/performance.md`, ADR status updates
- Record cold/warm parse, build, instantiate, solve, decode timings + end-to-end counts.
- Document where to add a DSL construct, rewrite pass, constraint, or backend.
- Update ADR-0005 status from Proposed to Accepted once the WMC backend lands.

**Acceptance:** `uv run pytest` passes; perf report shows no unapproved >5% regression.

---

## Migration Strategy

Adapter-first. Land the legacy adapter backend (Task B) and differential tests (Task C) before building
the native path, so every later change is validated against a working oracle. Introduce CellGraph IR and
the native backend behind the existing `Backend` protocol; never flag-day swap. Keep the bitset grounding
as a fallback rather than deleting it.

Recommended order: A → B → C → D → E → F → G → H → I → J → K.

## Risk Controls
- Keep every phase small enough to compare against baseline counts and timings.
- Add a benchmark (and a differential test) before every optimization so wins/regressions are measurable.
- Do not change numerical/ring backends in the same PR as structural refactors.
- Borrow ideas (B1–B9), not the reference's string-keyed atom representation.
