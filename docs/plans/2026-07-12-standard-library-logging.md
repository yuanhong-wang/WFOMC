# Standard-Library Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give the WFOMC library and CLI useful, default-silent diagnostic logging without retaining Loguru or adding a logging abstraction layer.

**Architecture:** Library modules create standard-library loggers with `logging.getLogger(__name__)` and never configure output handlers. The CLI alone maps `-v`/`-vv` to INFO/DEBUG and configures one concise stderr handler. Logs summarize phase boundaries, sizes, fallbacks, and timings; they never enumerate models, cells, pair factors, DP states, or complete grounded formulas.

**Tech Stack:** Python 3.11 standard-library `logging`, `time.perf_counter`, argparse, pytest, Ruff.

---

### Task 1: Logging contract and CLI verbosity

**Files:**
- Modify: `tests/unit/test_cli.py`
- Create: `tests/unit/test_logging_structure.py`
- Modify: `src/wfomc/cli.py`
- Modify: `src/wfomc/__init__.py`

**Steps:**

1. Add failing tests for `-v`/`-vv`, level mapping, a package `NullHandler`, and absence of Loguru imports.
2. Run the targeted tests and confirm they fail.
3. Add `-v/--verbose` to argparse and configure standard logging only from `cli.main()`.
4. Replace package-level Loguru disabling with a standard `NullHandler`.
5. Run the targeted tests and confirm they pass.

### Task 2: Migrate existing logs and make phase logs useful

**Files:**
- Modify: `src/wfomc/cell_graph/build.py`
- Modify: `src/wfomc/algo/propositional/counting.py`
- Modify: `src/wfomc/algo/propositional/ganak.py`
- Modify: `src/wfomc/algo/recursive/kernel.py`
- Modify: `src/wfomc/algo/recursive/solve.py`
- Test: `tests/unit/test_logging_structure.py`

**Steps:**

1. Replace every Loguru import with a module-local standard logger.
2. Replace `{}` formatting with lazy `%s` arguments.
3. Remove complete grounded-formula and GANAK-input logging.
4. Log cell graph counts and phase durations at INFO; keep predicate/branch details at DEBUG and fallback behavior at WARNING.
5. Log GANAK input size and execution duration without logging the full CNF.
6. Delete the unused recursive `PRINT_TREE`/`TreeNode` debug-print path rather than converting it into hot-loop logs.
7. Run algorithm and structure tests.

### Task 3: Remove dependency and document the boundary

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `docs/adr/0022-standard-library-logging.md`
- Modify: `docs/README.md`

**Steps:**

1. Remove Loguru from project dependencies and refresh the lockfile.
2. Add an ADR recording library/CLI ownership, level meanings, and prohibited high-volume logs.
3. Run Ruff lint and formatting checks for touched files.
4. Run `uv run pytest -q`.
5. Run the solver smoke matrix and `uv build`.

No staging or commit is included because this worktree contains the broader active architecture refactor.
