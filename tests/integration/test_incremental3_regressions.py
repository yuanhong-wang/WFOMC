"""End-to-end regressions specific to incremental3 counting semantics."""

from fractions import Fraction

import pytest

from wfomc import AlgoName, parse_problem, solve


def _binary_at_most_one_problem():
    return parse_problem(
        r"""
\forall X: (\exists_<=1 Y: R(X,Y))

domain = {a0, a1}

1 1 R
"""
    )


def test_incremental3_binary_le_counts_at_most_one_out_neighbor():
    assert solve(_binary_at_most_one_problem(), algo=AlgoName.INCREMENTAL3) == Fraction(
        9, 1
    )


def test_older_incremental_rejects_non_reducible_binary_le():
    with pytest.raises(ValueError, match="not reducible"):
        solve(_binary_at_most_one_problem(), algo=AlgoName.INCREMENTAL)


def test_incremental3_does_not_special_case_odd_and_u_predicate_names():
    problem = parse_problem(
        r"""
(\exists_=0 X: Odd(X)) & (\exists_=1 X: U(X))

domain = {a0, a1}

1 1 Odd
1 1 U
"""
    )

    assert solve(problem, algo=AlgoName.INCREMENTAL3) == Fraction(2, 1)
