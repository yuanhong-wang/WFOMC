# Domain-Free Boundary-Profile Tree Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build one reusable Boundary-Profile tree per structural input variant, add a reference-domain planning option to the Python API and CLI, and remove the obsolete shared master-sum module.

**Architecture:** The algorithm input template will own domain-free cell weights, pair weights, and a selected `BoundaryProfileTree`. Concrete-domain instantiation will only rebind arithmetic, materialize local tables, and derive estimates from the cached tree.

**Tech Stack:** Python 3.11, frozen dataclasses, pytest, the existing WFOMC staged engine and runtime cache.

---

### Task 1: Specify option and cache behavior with failing tests

**Files:**
- Modify: `tests/unit/test_engine_compile.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `tests/unit/test_boundary_profile_plan.py`

**Step 1: Write the failing input-template reuse test**

Instantiate one Boundary-Profile compilation for domains 2 and 3 with one
`RuntimeContext`. Assert one input-template cache miss, one hit, and object
identity of the component template's `tree` before and after both
instantiations.

**Step 2: Write failing option tests**

Assert that `BoundaryProfileOptions(tree_reference_domain_size=-1)` raises,
that two reference sizes produce distinct compilation cache entries, and that
the CLI accepts `--bp-tree-reference-domain-size` only for
`boundary-profile`.

**Step 3: Run tests to verify failure**

Run:

```bash
uv run pytest -q \
  tests/unit/test_engine_compile.py \
  tests/unit/test_cli.py \
  tests/unit/test_boundary_profile_plan.py
```

Expected: failures for the missing options and domain-free tree contract.

**Step 4: Commit the red tests**

```bash
git add tests/unit/test_engine_compile.py tests/unit/test_cli.py tests/unit/test_boundary_profile_plan.py
git commit -m "test: specify domain-free boundary-profile trees"
```

### Task 2: Add API and CLI planning options

**Files:**
- Modify: `src/wfomc/options.py`
- Modify: `src/wfomc/algo/core.py`
- Modify: `src/wfomc/algo/boundary_profile/spec.py`
- Modify: `src/wfomc/engine/orchestration.py`
- Modify: `src/wfomc/cli.py`
- Modify: `src/wfomc/__init__.py`
- Modify: `tests/unit/test_public_api.py`

**Step 1: Add the public options dataclass**

```python
@dataclass(frozen=True)
class BoundaryProfileOptions:
    tree_reference_domain_size: int | None = None

    def __post_init__(self) -> None:
        if (
            self.tree_reference_domain_size is not None
            and self.tree_reference_domain_size < 0
        ):
            raise ValueError("tree_reference_domain_size must be non-negative")
```

Add `boundary_profile_options` to `AlgoOptions`, preserve it in option
resolution, export it publicly, and include it in `_options_key`.

**Step 2: Add CLI parsing and scope validation**

Add a Boundary-Profile-only argument group and
`--bp-tree-reference-domain-size`. Convert it to `BoundaryProfileOptions` in
`run()` and reject it for every other algorithm.

**Step 3: Run option-focused tests**

Run:

```bash
uv run pytest -q tests/unit/test_cli.py tests/unit/test_public_api.py
```

Expected: all option and CLI tests pass; tree-cache tests remain red.

**Step 4: Commit the option layer**

```bash
git add src/wfomc tests/unit/test_cli.py tests/unit/test_public_api.py
git commit -m "feat: add boundary-profile tree planning option"
```

### Task 3: Split reusable tree topology from domain execution

**Files:**
- Modify: `src/wfomc/algo/boundary_profile/input.py`
- Modify: `src/wfomc/algo/boundary_profile/plan.py`
- Modify: `src/wfomc/algo/boundary_profile/kernel.py`
- Modify: `src/wfomc/algo/boundary_profile/solve.py`
- Delete: `src/wfomc/algo/master_sum.py`
- Modify: `tests/unit/test_boundary_profile_kernel.py`
- Modify: `tests/unit/test_boundary_profile_plan.py`
- Modify: `tests/unit/test_engine_compile.py`

**Step 1: Define algorithm-owned component types**

Move the scalar component and local-table materialization into
`boundary_profile/input.py`. The template component stores base weights, pair
weights, graph weight, and `BoundaryProfileTree`; the concrete component stores
`W_i(k)`, concrete pair weights, graph weight, and a materialized plan.

**Step 2: Build a domain-free tree**

Refactor the planner so candidate generation returns immutable topology.
Classify independent and symmetric nodes using base cell weights and diagonal
pair weights. Select with structural cost when the option is `None`, or with
the existing state-bound cost at the configured reference size.

**Step 3: Materialize one concrete plan without tree search**

Bind cross-interaction coordinates to concrete arithmetic values and calculate
the actual-domain estimates from the cached topology. Do not call candidate
generation during template instantiation.

**Step 4: Delete the obsolete shared module and update imports**

Remove `src/wfomc/algo/master_sum.py`; update kernel, planner, and tests to
import Boundary-Profile-owned contracts.

**Step 5: Run Boundary-Profile tests**

Run:

```bash
uv run pytest -q \
  tests/unit/test_boundary_profile_kernel.py \
  tests/unit/test_boundary_profile_plan.py \
  tests/unit/test_engine_compile.py \
  tests/integration/test_boundary_profile.py
```

Expected: all tests pass, including tree object reuse across domains.

**Step 6: Commit the planner refactor**

```bash
git add src/wfomc/algo tests/unit tests/integration/test_boundary_profile.py
git commit -m "refactor: cache boundary-profile trees across domains"
```

### Task 4: Verify correctness and architectural boundaries

**Files:**
- Modify: `README.md`
- Modify: `docs/adding-an-algorithm.md`
- Test: `tests/integration/test_boundary_profile.py`
- Test: `tests/unit/test_dependency_boundaries.py`

**Step 1: Document the option and cache semantics**

Describe the CLI/API option, default structural planning, reference-domain
planning, and one-tree-per-structural-variant guarantee.

**Step 2: Check for stale imports and whitespace errors**

Run:

```bash
rg -n "algo\.master_sum|MasterSum" src tests
git diff --check
```

Expected: no stale master-sum references and no whitespace errors.

**Step 3: Run the full test suite**

Run:

```bash
uv run pytest -q
```

Expected: the full suite passes with only existing skips.

**Step 4: Commit documentation and final verification updates**

```bash
git add README.md docs tests
git commit -m "docs: explain boundary-profile tree reuse"
```
