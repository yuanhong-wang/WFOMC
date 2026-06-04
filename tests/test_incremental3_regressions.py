from __future__ import annotations

import pytest

from wfomc import Algo, Const, Pred, Rational, WFOMCProblem, fol_parse, to_sc2, wfomc
from wfomc.algo.IncrementalWFOMC3 import ConfigSpace


def _domain(size: int) -> set[Const]:
    return {Const(f"a{i}") for i in range(size)}


def test_incremental3_binary_le_counts_at_most_one_out_neighbor() -> None:
    sentence = to_sc2(fol_parse(r"\forall X: (\exists_{<=1} Y: (R(X,Y)))"))
    problem = WFOMCProblem(
        sentence,
        _domain(2),
        {Pred("R", 2): (Rational(1, 1), Rational(1, 1))},
    )

    assert wfomc(problem, algo=Algo.INCREMENTAL3) == Rational(9, 1)


def test_binary_le_is_rejected_by_older_counting_encoding() -> None:
    sentence = to_sc2(fol_parse(r"\forall X: (\exists_{<=1} Y: (R(X,Y)))"))
    problem = WFOMCProblem(
        sentence,
        _domain(2),
        {Pred("R", 2): (Rational(1, 1), Rational(1, 1))},
    )

    with pytest.raises(RuntimeError, match="Binary counting comparator '<='"):
        wfomc(problem, algo=Algo.INCREMENTAL)


def test_incremental3_does_not_special_case_odd_and_u_predicate_names() -> None:
    sentence = to_sc2(
        fol_parse(r"\exists_{=0} X: (Odd(X)) & \exists_{=1} X: (U(X))")
    )
    problem = WFOMCProblem(
        sentence,
        _domain(2),
        {
            Pred("Odd", 1): (Rational(1, 1), Rational(1, 1)),
            Pred("U", 1): (Rational(1, 1), Rational(1, 1)),
        },
    )

    assert wfomc(problem, algo=Algo.INCREMENTAL3) == Rational(2, 1)


def test_incremental3_config_space_preserves_counts_above_uint8() -> None:
    space = ConfigSpace((1,))
    config = space.inc(space.zero, (0,), 256)

    config = space.inc(config, (0,))

    assert config[space.offset((0,))] == 257
