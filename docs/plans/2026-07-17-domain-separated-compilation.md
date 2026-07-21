# Domain-Separated Compilation Implementation Plan

**Goal:** Compile a logical WFOMC problem once and evaluate the compiled
artifact for several concrete domains, with Fast and FastV2 reusing their
domain-free reductions and cell-graph input templates.

**Architecture:** `Problem` and `Domain` are separate public values.
`CompiledProblem` is a reusable engine artifact, while `ProblemExecution` owns
one domain's arithmetic, algorithm inputs, decoder instances, and solver
caches. Algorithms use a staged contract for domain-free compilation, reusable
input templates, and concrete-domain instantiation.

**Tech Stack:** Python 3.11 frozen dataclasses, python-flint arithmetic,
pytest, Ruff.

---

### Task 1: Split source problems and domains

**Files:**

- Modify: `src/wfomc/problem.py`
- Modify: `src/wfomc/parser/`
- Modify: `src/wfomc/__init__.py`
- Test: `tests/unit/test_problem_parser.py`
- Test: `tests/unit/reduction/test_core_contract.py`

**Steps:**

1. Add immutable `Domain` and explicit `ProblemInstance`.
2. Remove `domain` and `circular_order_size` from `Problem`.
3. Give `Problem` and `Domain` independent stable cache keys.
4. Make parsers return `ProblemInstance(problem, domain)`.
5. Export the new public types and update parser contract tests.

### Task 2: Introduce reusable compilation and per-domain execution

**Files:**

- Add: `src/wfomc/engine/compilation.py`
- Modify: `src/wfomc/engine/orchestration.py`
- Modify: `src/wfomc/engine/runtime.py`
- Modify: `src/wfomc/engine/__init__.py`
- Modify: `src/wfomc/algo/core.py`
- Test: `tests/unit/test_engine_compile.py`
- Test: `tests/unit/test_engine.py`

**Steps:**

1. Define domain-free `CompiledProblem` and concrete `ProblemExecution`.
2. Move numeric compilation helpers from `algo.core` to engine compilation.
3. Add `instantiate_problem(compiled, domain)`.
4. Make `solve` accept a source problem plus domain, a parsed
   `ProblemInstance`, or a compiled problem plus domain.
5. Split runtime caches into compiled problems, input templates, bounded
   executions, and results.
6. Test that one compiled object is reused for different domains.

### Task 3: Add domain-free reduced problems

**Files:**

- Add: `src/wfomc/reduction/reduced.py`
- Modify: `src/wfomc/reduction/reduce_unary_evidence.py`
- Modify: `src/wfomc/reduction/reduce_counting_quantifiers.py`
- Modify: `src/wfomc/reduction/reduce_existential_quantifiers.py`
- Modify: `src/wfomc/reduction/reduce_cardinality_constraints.py`
- Modify: `src/wfomc/reduction/__init__.py`
- Test: `tests/unit/reduction/`

**Steps:**

1. Add `DomainExpr`, `ReducedProblem`, and data-only decoder specs.
2. Implement domain-free forms of the Fast/FastV2 reduction sequence.
3. Evaluate domain expressions, guards, profile capacities, and decoder
   coefficients during instantiation.
4. Retain concrete reductions only as low-level internal utilities.
5. Cover Fast input-structure selection at the open/closed unary-evidence
   boundary without creating separate logical reduction branches.

### Task 4: Stage Fast and FastV2 inputs

**Files:**

- Modify: `src/wfomc/algo/fast/graph.py`
- Modify: `src/wfomc/algo/fast/input.py`
- Modify: `src/wfomc/algo/fast/operations.py`
- Modify: `src/wfomc/algo/fast/spec.py`
- Modify: `src/wfomc/algo/fastv2/spec.py`
- Test: `tests/unit/test_engine_compile.py`
- Test: `tests/unit/test_cross_branch_performance.py`
- Test: `tests/unit/test_cell_graph_semantics.py`

**Steps:**

1. Build reusable weighted cell-graph data from a compiled branch.
2. Compute a static clique layout without a domain-size loop.
3. Store that layout in `FastInputTemplate`.
4. Rebind input-template values into concrete arithmetic and construct fresh
   operations/caches for each domain.
5. Keep the existing Fast solver hot loop unchanged.
6. Verify Fast and FastV2 answers against the standard algorithm.

### Task 5: Migrate remaining algorithms and callers

**Files:**

- Modify: `src/wfomc/algo/*/spec.py`
- Modify: `src/wfomc/reduction/core.py`
- Modify: `src/wfomc/cli.py`
- Modify: `tests/`
- Modify: `benchmarks/`

**Steps:**

1. Move remaining algorithms onto the staged input-template contract.
2. Delete the superseded concrete reduction stage types and wrappers.
3. Update CLI and benchmark call sites to pass parsed `ProblemInstance`.
4. Update direct `Problem(...)` construction to construct or pass a `Domain`
   separately.
5. Remove the old domain-sensitive `Problem.cache_key_parts` behavior.

### Task 6: Verify behavior and reuse

**Steps:**

1. Run focused problem/parser, reduction, engine, Fast, and FastV2 tests.
2. Run the complete test suite.
3. Run Ruff over source, tests, and benchmarks.
4. Run the package build and CLI smoke tests.
5. Run `git diff --check`.
6. Confirm cache statistics show one compiled/input-template build and separate
   execution/result entries for different domains.
