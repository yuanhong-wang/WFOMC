"""Comparator coverage for incremental3 counting automata."""

from __future__ import annotations

from pathlib import Path

import pytest

from wfomc import (
    AlgoName,
    compile_problem,
    instantiate_problem,
    parse_problem_file,
    parse_problem,
    solve,
)


ROOT = Path(__file__).parents[2]
COUNTING_MODEL_DIR = ROOT / "models" / "counting_quantifiers"
COUNTING_MODEL_CASES = (
    ("global-eq.wfomcs", 3),
    ("global-le.wfomcs", 7),
    ("global-lt.wfomcs", 4),
    ("global-ge.wfomcs", 4),
    ("global-gt.wfomcs", 1),
    ("global-ne.wfomcs", 5),
    ("row-eq.wfomcs", 4),
    ("row-le.wfomcs", 9),
    ("row-lt.wfomcs", 1),
    ("row-ge.wfomcs", 9),
    ("row-gt.wfomcs", 1),
    ("row-ne.wfomcs", 4),
    ("nullary-definition-eq.wfomcs", 21),
    ("nullary-definition-le.wfomcs", 17),
    ("nullary-definition-lt.wfomcs", 20),
    ("nullary-definition-ge.wfomcs", 20),
    ("nullary-definition-gt.wfomcs", 23),
    ("nullary-definition-ne.wfomcs", 19),
    ("unary-definition-eq.wfomcs", 100),
    ("unary-definition-le.wfomcs", 81),
    ("unary-definition-lt.wfomcs", 121),
    ("unary-definition-ge.wfomcs", 81),
    ("unary-definition-gt.wfomcs", 121),
    ("unary-definition-ne.wfomcs", 100),
    ("embedded-row-count.wfomcs", 25),
    ("mod-count-definitions.wfomcs", 168_000),
    ("existential-as-ge-one.wfomcs", 9),
    ("facility-location.wfomcs", 120),
)


GLOBAL_CASES = (
    ("=", 2, 3),
    ("<=", 2, 7),
    ("<", 2, 4),
    (">=", 2, 4),
    (">", 2, 1),
    ("!=", 2, 5),
)

ROW_CASES = (
    ("=", 1, 2),
    ("<=", 1, 3),
    ("<", 1, 1),
    (">=", 1, 3),
    (">", 1, 1),
    ("!=", 1, 2),
)

BOUNDARY_CASES = (
    ("<", 0, 0, 0),
    (">=", 0, 1, 1),
    ("=", 3, 0, 0),
    ("!=", 3, 4, 16),
    (">", 2, 0, 0),
    ("<=", 2, 4, 16),
)


@pytest.mark.parametrize(("filename", "expected"), COUNTING_MODEL_CASES)
def test_incremental3_counting_model_corpus(filename: str, expected: int):
    problem = parse_problem_file(str(COUNTING_MODEL_DIR / filename))

    assert solve(problem, algo=AlgoName.INCREMENTAL3) == expected


@pytest.mark.parametrize(("comparator", "threshold", "models"), GLOBAL_CASES)
def test_incremental3_supports_all_global_count_comparators(
    comparator: str,
    threshold: int,
    models: int,
):
    problem = parse_problem(
        rf"""
\exists_{{{comparator}{threshold}}} X: U(X)
domain = 3
"""
    )

    assert solve(problem, algo=AlgoName.INCREMENTAL3) == models


@pytest.mark.parametrize(("comparator", "threshold", "row_models"), ROW_CASES)
def test_incremental3_supports_all_row_count_comparators(
    comparator: str,
    threshold: int,
    row_models: int,
):
    problem = parse_problem(
        rf"""
\forall X: (\exists_{{{comparator}{threshold}}} Y: R(X,Y))
domain = 2
"""
    )

    assert solve(problem, algo=AlgoName.INCREMENTAL3) == row_models**2


@pytest.mark.parametrize(("comparator", "threshold", "models"), GLOBAL_CASES)
def test_incremental3_supports_nullary_count_definitions(
    comparator: str,
    threshold: int,
    models: int,
):
    problem = parse_problem(
        rf"""
A <-> (\exists_{{{comparator}{threshold}}} X: U(X))
domain = 3
2 3 A
"""
    )

    expected = 2 * models + 3 * (2**3 - models)
    assert solve(problem, algo=AlgoName.INCREMENTAL3) == expected


@pytest.mark.parametrize(("comparator", "threshold", "row_models"), ROW_CASES)
def test_incremental3_supports_unary_count_definitions(
    comparator: str,
    threshold: int,
    row_models: int,
):
    problem = parse_problem(
        rf"""
\forall X: (A(X) <-> (\exists_{{{comparator}{threshold}}} Y: R(X,Y)))
domain = 2
2 3 A
"""
    )

    expected_per_row = 2 * row_models + 3 * (2**2 - row_models)
    assert solve(problem, algo=AlgoName.INCREMENTAL3) == expected_per_row**2


@pytest.mark.parametrize(
    ("comparator", "threshold", "global_models", "row_models"),
    BOUNDARY_CASES,
)
def test_incremental3_handles_count_boundaries(
    comparator: str,
    threshold: int,
    global_models: int,
    row_models: int,
):
    global_problem = parse_problem(
        rf"\exists_{{{comparator}{threshold}}} X: U(X)" "\ndomain = 2"
    )
    row_problem = parse_problem(
        rf"\forall X: (\exists_{{{comparator}{threshold}}} Y: R(X,Y))"
        "\n"
        "domain = 2"
    )

    assert solve(global_problem, algo=AlgoName.INCREMENTAL3) == global_models
    assert solve(row_problem, algo=AlgoName.INCREMENTAL3) == row_models


def test_incremental3_reuses_one_projection_for_multiple_counters():
    problem = parse_problem(
        r"""
(\forall X: (\exists_>=1 Y: R(X,Y))) &
(\forall X: (\exists_<=1 Y: R(X,Y)))
domain = 2
"""
    )

    assert solve(problem, algo=AlgoName.INCREMENTAL3) == 4


def test_incremental3_supports_negated_global_count_definition():
    problem = parse_problem(
        r"""
~(\exists_!=2 X: U(X))
domain = 3
"""
    )

    assert solve(problem, algo=AlgoName.INCREMENTAL3) == 3


def test_incremental3_supports_row_count_inside_disjunction():
    problem = parse_problem(
        r"""
\forall X: (P(X) | (\exists_=2 Y: R(X,Y)))
domain = 2
"""
    )

    # For each row: P=true permits four R rows; P=false permits only all-true.
    assert solve(problem, algo=AlgoName.INCREMENTAL3) == 5**2


def test_incremental3_supports_mod_count_definitions():
    global_problem = parse_problem(
        r"""
A <-> (\exists_{1mod3} X: U(X))
domain = 3
2 3 A
"""
    )
    row_problem = parse_problem(
        r"""
\forall X: (A(X) <-> (\exists_{1mod2} Y: R(X,Y)))
domain = 2
2 3 A
"""
    )

    assert solve(global_problem, algo=AlgoName.INCREMENTAL3) == 2 * 3 + 3 * 5
    assert solve(row_problem, algo=AlgoName.INCREMENTAL3) == (2 * 2 + 3 * 2) ** 2


def test_incremental3_collapses_count_threshold_above_domain_size():
    impossible = parse_problem(
        r"""
\forall X: (\exists_=1000000 Y: R(X,Y))
domain = 2
"""
    )
    non_equal = parse_problem(
        r"""
\forall X: (\exists_!=1000000 Y: R(X,Y))
domain = 2
"""
    )
    large_modulus = parse_problem(
        r"""
\forall X: (\exists_{1mod1000000} Y: R(X,Y))
domain = 2
"""
    )

    artifacts = instantiate_problem(
        compile_problem(impossible.problem, algo=AlgoName.INCREMENTAL3),
        impossible.domain,
    )
    mod_artifacts = instantiate_problem(
        compile_problem(large_modulus.problem, algo=AlgoName.INCREMENTAL3),
        large_modulus.domain,
    )

    assert artifacts.algo_input.counting_state.c_type_shape == (1,)
    assert mod_artifacts.algo_input.counting_state.c_type_shape == (2,)
    assert solve(impossible, algo=AlgoName.INCREMENTAL3) == 0
    assert solve(non_equal, algo=AlgoName.INCREMENTAL3) == 2 ** (2 * 2)
    assert solve(large_modulus, algo=AlgoName.INCREMENTAL3) == 2**2
