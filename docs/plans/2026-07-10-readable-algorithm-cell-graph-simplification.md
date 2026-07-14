# Readable Algorithm and Cell Graph Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the WFOMC execution path readable from algorithm entry to solver while reducing `cell_graph` to shared 1-type/2-type data and fixing the correctness and public-contract failures identified in the architecture review.

**Architecture:** Keep a modular monolith and a small algorithm registry. A shared cell-graph builder produces plain `CellGraphData`; algorithm packages explicitly derive fast clique state, ordered pair tables, counting transitions, ground CNF, or tail-signature tables. Do not add generic Raw/Reduced/Materialized object hierarchies, visitor layers, or new plugin frameworks.

**Tech Stack:** Python 3.11, dataclasses, python-flint, Lark, pytest, Ruff.

---

## Implementation status

Implemented in the current worktree:

- correctness/public-contract guardrails;
- `CellGraphData` and table-only algorithm migration;
- algorithm-owned ordered, counting, CNF, recursive, tail, and fast input preparation;
- deletion of the shared algorithm adapter modules and mirrored input fields;
- explicit `AlgoSpec.prepare` / `AlgoSpec.solve` orchestration;
- FOL dependency-direction fix, stable CLI, README, ADR, benchmark cleanup, and CI checks.

Fast clique discovery, optimized recurrences, and materialized operations now
live in `algo/fast/graph.py`, `algo/fast/weights.py`, and
`algo/fast/operations.py`. The shared `cell_graph.py` contains only the base
builder, plain-data snapshot, nullary branching, and a lazy dispatch used by
the fast preparation path.

## Constraints

- Preserve current exact counts for supported inputs.
- Unsupported inputs must fail before solving; never silently ignore a feature.
- Prefer explicit algorithm preparation code and small data records over callable wrappers.
- Allow small duplication when it makes an algorithm readable in one package.
- Preserve unrelated uncommitted work. Do not commit or rewrite user changes.
- Migrate one dependency family at a time and run focused tests after every task.

### Task 1: Pin correctness and public-contract guardrails

**Files:**
- Modify: `tests/unit/test_feature_analysis.py`
- Modify: `tests/unit/test_engine_compile.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `tests/unit/test_weight_backend_planning.py`
- Modify: `src/wfomc/engine/features.py`
- Modify: `src/wfomc/algo/core.py`
- Modify: `pyproject.toml`

**Steps:**

1. Add a failing test proving a `Problem` with binary evidence is rejected rather than counted unchanged.
2. Add `FeatureSet.has_binary_evidence` and validate it centrally with `UnsupportedFeatureError` until an algorithm explicitly supports it.
3. Add a console-entry test that resolves `wfomc.cli:main`; change the `wfomc` script to that entry and remove `new_wfomc`.
4. Add a test that `round/float` fails during option resolution with a clear unsupported-backend error; do not allow the current mixed `float`/`fmpq` failure in the solver.
5. Run:

   ```bash
   uv run pytest -q tests/unit/test_feature_analysis.py tests/unit/test_engine_compile.py tests/unit/test_cli.py tests/unit/test_weight_backend_planning.py
   ```

   Expected: all focused tests pass.

### Task 2: Introduce the minimal shared cell-graph data boundary

**Files:**
- Modify: `src/wfomc/cell_graph/components.py`
- Modify: `src/wfomc/cell_graph/cell_graph.py`
- Modify: `src/wfomc/cell_graph/__init__.py`
- Create: `tests/unit/test_cell_graph_data.py`

**Target data:**

```python
@dataclass(frozen=True)
class CellGraphData:
    cells: tuple[Cell, ...]
    cell_weights: tuple[object, ...]
    two_tables: tuple[tuple[TwoTable, ...], ...]
    predecessor_two_tables: tuple[tuple[int, tuple[tuple[TwoTable, ...], ...]], ...] = ()

    def pair_weight(self, left: int, right: int, evidence=None) -> object:
        return self.two_tables[left][right].get_weight(evidence)

    def predecessor_pair_weight(self, order: int, left: int, right: int) -> object:
        ...
```

**Steps:**

1. Add tests comparing `CellGraphData` cells, unconditional pair weights, conditional pair weights, and predecessor weights with the current `CellGraph` behavior.
2. Add one `snapshot()` operation to the current builder; keep construction behavior unchanged initially.
3. Make the non-optimized `build_cell_graphs` path yield `CellGraphData` plus nullary branch weight.
4. Keep `Cell` and `TwoTable` as small domain objects; do not wrap them in additional interfaces.
5. Run:

   ```bash
   uv run pytest -q tests/unit/test_cell_graph_data.py tests/unit/test_engine_compile.py tests/unit/test_normal_form.py
   ```

### Task 3: Migrate table-only algorithms first

**Files:**
- Modify: `src/wfomc/algo/standard/input.py`
- Modify: `src/wfomc/algo/standard/solve.py`
- Modify: `src/wfomc/algo/recursive/spec.py`
- Modify: `src/wfomc/algo/recursive/solve.py`
- Modify: `src/wfomc/algo/tail_signature/spec.py`
- Modify: `src/wfomc/algo/tail_signature/solve.py`
- Modify: `src/wfomc/algo/cell_graph/types.py`
- Modify: `src/wfomc/algo/cell_graph/cache.py`
- Test: `tests/test_formula_models.py`
- Test: `tests/unit/test_tail_signature_runtime.py`

**Steps:**

1. Replace live-graph extraction with direct `CellGraphData` reads in standard.
2. Make recursive preparation create its input directly; remove its dependency on the generic `build_basic_cell_graph_input_from_reduced` wrapper.
3. Make tail-signature preparation build `w_tables`/`r_matrix` directly from `CellGraphData`, avoiding the intermediate `BasicCellGraphInput`.
4. Remove the first-component mirror fallback from migrated solvers; `components` is the only source of truth.
5. Run focused algorithm parity tests, followed by:

   ```bash
   uv run pytest -q tests/test_formula_models.py tests/unit/test_tail_signature_runtime.py
   ```

### Task 4: Move ordered and counting derivations into their algorithms

**Files:**
- Modify: `src/wfomc/algo/incremental/input.py`
- Modify: `src/wfomc/algo/incremental/spec.py`
- Modify: `src/wfomc/algo/incremental/solve.py`
- Modify: `src/wfomc/algo/incremental3/input.py`
- Modify: `src/wfomc/algo/incremental3/spec.py`
- Modify: `src/wfomc/algo/incremental3/solve.py`
- Modify: `src/wfomc/algo/incremental3/counting_kernel.py`
- Modify: `src/wfomc/algo/cell_graph/components.py`
- Modify: `src/wfomc/algo/cell_graph/inputs.py`
- Test: `tests/unit/test_incremental3_counting_native.py`
- Test: `tests/test_incremental3_regressions.py`

**Steps:**

1. Add `incremental.prepare` code that converts `CellGraphData.predecessor_two_tables` into the exact PREDk/circular matrices consumed by its solver.
2. Delete unused `leq_pair_tables`, input-level mirrored tables, and `predecessor_pair_tables` aliases.
3. Add `incremental3.prepare` code that builds initial states and conditional binary transition weights using `CellGraphData.pair_weight(..., evidence)`.
4. Remove the obsolete `counting_kernel.build_weight(cells, live_graph, state)` path.
5. Keep evidence allocation separate from graph data and attach it in each algorithm's prepare function.
6. Run:

   ```bash
   uv run pytest -q tests/unit/test_incremental3_counting_native.py tests/test_incremental3_regressions.py tests/unary_evidence
   ```

### Task 5: Isolate fast/fastv2 optimization from the shared graph

**Files:**
- Create: `src/wfomc/algo/fast/prepare.py`
- Modify: `src/wfomc/algo/fast/input.py`
- Modify: `src/wfomc/algo/fast/spec.py`
- Modify: `src/wfomc/algo/fastv2/spec.py`
- Modify: `src/wfomc/algo/fast/solve.py`
- Modify: `src/wfomc/cell_graph/cell_graph.py`
- Delete: `src/wfomc/cell_graph/materialized.py`
- Modify: `src/wfomc/cell_graph/optimized_weights.py`
- Test: `tests/unary_evidence/test_algorithm_matrix.py`
- Test: `tests/test_formula_models.py`

**Steps:**

1. Write parity tests for fast and fastv2 on plain, cardinality, and supported evidence cases.
2. Move clique matching, independent-set discovery, and fast-specific `J/d/term` state into `algo/fast/prepare.py`.
3. Construct one `FastComponent` containing the base matrices and fast-specific structure; eliminate the live `OptimizedCellGraph -> MaterializedOptimizedOperations` snapshot chain.
4. Keep one implementation of `J/d/term` recursion in the fast package.
5. Move lifted-profile expansion, if still supported by fast, into fast preparation; do not make it a shared cell node type.
6. Delete `OptimizedCellGraph`, `OptimizedCellGraphWithEvidence`, `CellWithEvidenceProfile`, and `cell_graph/materialized.py` after all reads are gone.
7. Run:

   ```bash
   uv run pytest -q tests/unary_evidence/test_algorithm_matrix.py tests/test_formula_models.py
   ```

### Task 6: Remove the generic cell-graph adapter layer

**Files:**
- Delete: `src/wfomc/algo/cell_graph/components.py`
- Delete: `src/wfomc/algo/cell_graph/inputs.py`
- Simplify: `src/wfomc/algo/cell_graph/types.py`
- Modify: all algorithm `spec.py` and `input.py` files still importing these modules
- Modify: `src/wfomc/algo/cell_graph/cache.py`

**Steps:**

1. Move the small shared cache beside `cell_graph.build_cell_graphs`; cache only `CellGraphData` branches.
2. Let each algorithm's prepare function explicitly build its own input from the shared data.
3. Remove cross-imports where the shared cell-graph package imports concrete fast/incremental/incremental3 input types.
4. Verify no production import remains:

   ```bash
   rg "algo\.cell_graph\.(components|inputs)" src tests
   ```

   Expected: no output.

### Task 7: Flatten algorithm dispatch and materialization wrappers

**Files:**
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/engine/orchestration.py`
- Modify: every `src/wfomc/algo/*/spec.py`
- Modify: `tests/unit/engine/test_orchestration.py`
- Modify: `tests/unit/test_engine_compile.py`

**Target contract:**

```python
@dataclass(frozen=True)
class AlgoSpec:
    name: AlgoName
    prepare: Callable[[Problem, AlgoOptions, RuntimeContext], tuple[PreparedBranch, ...]]
    solve: Callable[[AlgoInput, RuntimeContext], WFOMCResult]
```

`PreparedBranch` is allowed only if real reduction branching remains; otherwise return one input plus decoder directly. Do not add a generic materialization context or visitor.

**Steps:**

1. Add orchestration tests that pin `analyze -> prepare -> solve -> decode` and multi-branch decoder order.
2. Inline `backend_materializer`, `ufo2_materializer`, and `counting_dp_materializer` behavior into explicit algorithm preparation functions.
3. Remove the separate callable tuple of reductions from `AlgoSpec`; each prepare function lists its reduction calls in readable order.
4. Keep engine responsible only for feature analysis, cache lookup, prepare invocation, solve invocation, and branch result aggregation.
5. Run engine and algorithm matrix tests.

### Task 8: Fix dependency direction, documentation, benchmark, and CI drift

**Files:**
- Modify: `src/wfomc/fol/grounding.py`
- Modify: `src/wfomc/cell_graph/formula_ops.py`
- Modify: `README.md`
- Modify: `docs/README.md`
- Create: `docs/adr/0007-plain-cell-graph-and-explicit-algorithm-preparation.md`
- Modify or delete: `benchmarks/run_framework_benchmarks.py`
- Modify: `benchmarks/run_tail_signature.py`
- Modify: `benchmarks/README.md`
- Modify: `.github/workflows/python-app.yml`

**Steps:**

1. Move generic grounding helpers into `fol`; ensure `fol` no longer imports `cell_graph`.
2. Update README to the real `wfomc` CLI and current Python API.
3. Record the accepted plain-data cell graph and explicit prepare decision as ADR-0007.
4. Repair benchmark imports or remove entrypoints whose implementation no longer exists.
5. Add CI steps for Ruff, wheel build/install, `wfomc --help`, and one small model solve.
6. Run dependency and documentation checks.

### Task 9: Final verification and dead-code deletion

**Steps:**

1. Run:

   ```bash
   uv run ruff check src tests benchmarks
   uv run pytest -q
   uv build
   uv run wfomc --help
   uv run wfomc -i models/2-colored-graph.wfomcs --algo standard
   ```

2. Search for retired abstractions and mirror fields.
3. Compare representative results across standard, fast, fastv2, incremental, incremental3, recursive, and propositional where the external backend is available.
4. Update the architecture review with completion status and remaining unsupported features.

## Failure and rollback strategy

- If a migrated algorithm changes a count, keep the previous path temporarily behind a test-only comparison helper; do not add a public compatibility flag.
- If fast parity cannot be established, stop after Task 4 with base, ordered, and counting algorithms simplified; do not partially delete optimized graph code.
- If an existing dirty-worktree edit conflicts with a task, preserve it and adapt the task locally rather than restoring files from HEAD.
