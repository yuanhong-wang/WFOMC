from __future__ import annotations

import pytest

from types import SimpleNamespace

from wfomc import (
    Algo,
    Const,
    Counting,
    Pred,
    QuantifiedFormula,
    Rational,
    Universal,
    Var,
    WFOMCProblem,
    X,
    Y,
    fol_parse,
    to_sc2,
    wfomc,
)
from wfomc.algo.IncrementalWFOMC3 import ConfigSpace, build_weight
from wfomc.context import CountingState, IncrementalWFOMC3Context, UnaryConstraintHandler
from wfomc.parser.wfomcs_parser import parse as wfomcs_parse


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


def test_wfomcs_parse_returns_problem_object() -> None:
    problem = wfomcs_parse(r"\forall X: (P(X))" "\n" "V = 1")

    assert isinstance(problem, WFOMCProblem)


def test_counting_quantifier_rejects_invalid_inputs_without_asserts() -> None:
    with pytest.raises(ValueError, match="Unsupported comparator"):
        Counting(Var("X"), "??", 1)

    with pytest.raises(ValueError, match="Require 0 ≤ r < k"):
        Counting(Var("X"), "mod", (0, 0))


def test_incremental3_rejects_non_atomic_binary_counting_bodies() -> None:
    ctx = IncrementalWFOMC3Context.__new__(IncrementalWFOMC3Context)
    ctx.unary_handler = UnaryConstraintHandler()
    ctx._cnt_preds = []
    ctx._cnt_params = []
    ctx._cnt_remainder = []
    ctx._mod_pred_index = []
    ctx._exist_mod = False
    ctx._exist_le = False
    ctx._le_pred = []
    ctx._comparator_handlers = {
        "mod": ctx._handle_mod,
        "=": ctx._handle_eq,
        "<=": ctx._handle_le,
    }
    ctx.sentence = SimpleNamespace(
        cnt_formulas=[
            QuantifiedFormula(
                Universal(X),
                QuantifiedFormula(
                    Counting(Y, "<=", 1),
                    Pred("R", 2)(X, Y) | Pred("S", 2)(X, Y),
                ),
            )
        ]
    )

    with pytest.raises(TypeError, match="Binary counting quantifier requires a binary atomic formula"):
        ctx._handle_counting_quantifier()


def test_build_weight_wraps_modulo_remainder_for_positive_cells() -> None:
    class _Cell:
        def __init__(self, positive: bool):
            self.positive = positive

        def is_positive(self, _pred):
            return self.positive

    class _CellGraph:
        @staticmethod
        def get_cell_weight(_cell):
            return Rational(1, 1)

        @staticmethod
        def get_two_table_weight(_pair, _evidence):
            return Rational(0, 1)

    state = CountingState(
        ext_preds=[],
        cnt_preds=[Pred("R", 1)],
        cnt_params=[2],
        cnt_remainder=[0],
        exist_mod=True,
        mod_pred_index=[0],
        exist_le=False,
        le_index=[],
        binary_evidence=[],
        c_type_shape=[2],
    )

    w2t, _, _ = build_weight([_Cell(True)], _CellGraph(), state)

    assert w2t[0] == (1,)
