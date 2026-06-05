"""Regression test for ganak_count on a degenerate (empty) CNF.

Does not require the ganak binary: the empty-CNF case short-circuits before the
binary is located.
"""
from wfomc.algo.ganak import ganak_count


def test_ganak_count_empty_cnf_returns_one():
    # No propositional variables -> exactly one (empty) model with weight 1.
    # ganak asserts on empty input, so ganak_count must short-circuit.
    assert ganak_count(0, [], {}) == 1
