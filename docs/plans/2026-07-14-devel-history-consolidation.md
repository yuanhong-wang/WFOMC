# Devel History Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the 114-commit architecture-refactor history with a small,
reviewable set of thematic commits based on `devel@2ca2b38`, without changing
the final repository tree except for explicit Markdown whitespace cleanup.

**Architecture:** First commit the complete green working tree as a recoverable
snapshot and retain it under a backup branch. Then mixed-reset the active
branch to `devel`, rebuild the same tree in five path-oriented commits, verify
that the consolidated tip differs from the snapshot only by the recorded
Markdown whitespace cleanup, and finally promote the active local branch to
`devel`. No remote refs are changed.

**Tech Stack:** Git, Python 3.11, pytest, uv.

---

### Task 1: Protect the current state

**Files:**
- Snapshot: all tracked and untracked project files

**Steps:**

1. Confirm `devel`, `origin/devel`, and `origin/modk` all point to
   `2ca2b381941d76f51956daa528717e329aff8b87`.
2. Confirm there are no staged or unmerged files and inspect untracked paths
   for credentials or generated artifacts.
3. Run the complete test suite with a five-minute hard timeout.
4. Stage the complete working tree and create a temporary snapshot commit.
5. Create `backup/modk-framework-pre-consolidation-20260714` at the snapshot
   commit and record its tree hash.

### Task 2: Reset the active history without changing files

**Files:**
- Modify: Git index and active branch only

**Steps:**

1. Run a mixed reset of `codex/modk-framework` to local `devel`.
2. Confirm the active HEAD is exactly `2ca2b38` and the final files remain in
   the working tree.
3. Confirm the expected final diff is still present and there are no conflict
   entries.

### Task 3: Create the consolidated commits

**Files:**
- Commit 1: `src/`, `pyproject.toml`, `uv.lock`
- Commit 2: `tests/`, `.github/workflows/`
- Commit 3: `models/`
- Commit 4: `benchmarks/`, `scripts/`
- Commit 5: `README.md`, `docs/`, `plan/`

**Steps:**

1. Commit the typed FOL, problem, reduction, cell-graph, algorithm, arithmetic,
   evidence, parser, engine, and public API implementation as the core rewrite.
2. Commit the reorganized unit/integration test suite and CI configuration.
3. Commit the reorganized model corpus.
4. Commit the reusable benchmark catalog and benchmark/tool scripts.
5. Commit architecture reviews, ADRs, experiment reports, migration plans, and
   top-level documentation.
6. Confirm no project file remains unstaged or untracked.

### Task 4: Prove the rewrite preserved the final state

**Files:**
- Verify: complete repository

**Steps:**

1. Compare the consolidated tip with the backup snapshot; only the three
   explicitly cleaned Markdown plan files and this plan's recorded exception
   may differ.
2. Run the complete test suite with a five-minute hard timeout.
3. Run `compileall`, `uv build`, and `git diff --check`.
4. Confirm the branch is five commits ahead of `devel@2ca2b38` and the working
   tree is clean.

### Task 5: Promote the local branch to devel

**Files:**
- Modify: local branch refs only

**Steps:**

1. Delete the old local `devel` ref, which is recoverable from `origin/devel`.
2. Rename `codex/modk-framework` to `devel`.
3. Set its upstream to `origin/devel` and confirm it is five commits ahead.
4. Keep the backup branch locally and do not push any remote ref.
