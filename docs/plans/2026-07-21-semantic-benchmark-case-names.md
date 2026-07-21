# Semantic Benchmark Case Names Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Directly migrate benchmark case IDs and families from historical shorthand to precise semantic names.

**Architecture:** Rename catalog families and construct keys from semantic problem plus encoding variant. Preserve formulas, suites, domains, and correction metadata; update only active code/tests/docs, not historical result artifacts.

**Tech Stack:** Python 3.11, dataclasses, pytest, existing WFOMC benchmark catalog.

---

### Task 1: Lock the semantic catalog contract

**Files:**
- Modify: `tests/unit/test_benchmark_cases.py`
- Modify: `tests/unit/test_cross_branch_performance.py`
- Modify: `tests/unit/test_domain_series_performance.py`

1. Replace old keys in tests with semantic keys.
2. Assert the complete family/variant inventory and rejection of representative old keys.
3. Run focused tests and confirm they fail against the old catalog.

### Task 2: Migrate catalog names

**Files:**
- Modify: `benchmarks/cases.py`

1. Rename private definition/build helpers to semantic terms.
2. Replace core family names and generated keys.
3. Represent C2/cardinality encodings as variants of semantic families.
4. Rename unary families to include the constrained predicate.
5. Rename comparison groups without historical `hand` terminology.
6. Run focused tests.

### Task 3: Document and verify

**Files:**
- Modify: `benchmarks/README.md`

1. Document the semantic key structure and direct migration.
2. Search active code/tests for removed identifiers.
3. Run compile checks, focused benchmark tests, full pytest, and a catalog smoke benchmark.
4. Commit the migration.
