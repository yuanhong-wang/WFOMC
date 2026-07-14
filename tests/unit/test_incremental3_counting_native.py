from __future__ import annotations

import pytest

from wfomc.algo import AlgoName, AlgoOptions, ExistentialStrategy
from wfomc.algo.incremental3.counting_kernel import ConfigSpace
from wfomc.algo.incremental3.counting_state import CountingState
from wfomc.algo.incremental3.input import _initial_state
from wfomc.cell_graph import Cell
from wfomc.engine import compile_problem, solve
from wfomc.fol import Predicate
from wfomc.parser import parse_input, parse_problem
from wfomc.errors import UnsupportedFeatureError


def test_config_space_preserves_counts_above_uint8():
    space = ConfigSpace((1,))
    config = space.inc(space.zero, (0,), 256)

    config = space.inc(config, (0,))

    assert config[space.offset((0,))] == 257


def test_positive_mod_cell_wraps_initial_remainder():
    predicate = Predicate("Remainder", 1)
    state = CountingState(
        ext_preds=(),
        cnt_preds=(predicate,),
        cnt_params=(2,),
        cnt_remainder=(0,),
        exist_mod=True,
        mod_pred_index=(0,),
        exist_le=False,
        le_index=(),
        binary_evidence=(),
        c_type_shape=(2,),
    )

    assert _initial_state(Cell((True,), (predicate,)), state) == (1,)
    assert _initial_state(Cell((False,), (predicate,)), state) == (0,)


def test_incremental3_materialization_carries_counting_state():
    artifacts = compile_problem(
        parse_input("models/modk/1mod2-regular-graph.wfomcs"),
        algo=AlgoName.INCREMENTAL3,
    )
    state = artifacts.algo_input.counting_state
    masks = artifacts.algo_input.unary_cardinality_masks

    assert state.exist_mod is True
    assert state.cnt_params == (2,)
    assert state.cnt_remainder == (1,)
    assert len(state.cnt_preds) == 1
    assert masks.mod_constraints == []
    assert masks.eq_constraints == []
    assert masks.le_constraints == []
    assert masks.ge_constraints == []


def test_global_count_predicate_is_materialized_without_formula_patch():
    problem = parse_problem(
        r"""
\exists_=1 X: U(X)
domain = 2
"""
    )

    artifacts = compile_problem(problem, algo=AlgoName.INCREMENTAL3)
    normal_form = artifacts.reduced_problem.expect_single().problem.normal_form
    masks = artifacts.algo_input.unary_cardinality_masks

    assert normal_form.qf_formula is None
    assert [(predicate.name, count) for predicate, count in masks.eq_constraints] == [
        ("U", 1)
    ]
    assert len(artifacts.algo_input.components[0].cells) == 2
    assert solve(problem, algo=AlgoName.INCREMENTAL3) == 2


def test_incremental3_defaults_existentials_to_counting_sections():
    problem = parse_problem(
        r"""
\forall X: (\exists Y: (P(Y) & R(X,Y)))
domain = 2
"""
    )

    artifacts = compile_problem(problem, algo=AlgoName.INCREMENTAL3)
    normal_form = artifacts.reduced_problem.problems[0].problem.normal_form

    assert artifacts.algo_options.existential_strategy is ExistentialStrategy.COUNTING
    assert normal_form.forall_exists == ()
    assert len(normal_form.forall_counts) == 1
    assert normal_form.forall_counts[0].comparator == ">="
    assert normal_form.forall_counts[0].count == 1
    assert normal_form.forall_counts[0].body.predicate.name.startswith(
        "__existential_count"
    )


def test_other_algorithms_reject_counting_existential_strategy():
    problem = parse_problem(
        r"""
\forall X: (\exists Y: (R(X,Y)))
domain = 2
"""
    )

    with pytest.raises(UnsupportedFeatureError, match="existential strategy"):
        solve(
            problem,
            algo=AlgoName.FASTV2,
            options=AlgoOptions(
                existential_strategy=ExistentialStrategy.COUNTING,
            ),
        )


def test_counting_and_skolem_existential_strategies_are_exact():
    cases = (
        (r"\forall X: (\exists Y: (R(X,Y)))", 9),
        (r"\forall X: (\exists Y: (P(Y) & R(X,Y)))", 17),
        (r"\exists X: (P(X))", 3),
    )

    for sentence, expected in cases:
        problem = parse_problem(f"{sentence}\ndomain = 2")
        for strategy in (ExistentialStrategy.COUNTING, ExistentialStrategy.SKOLEM):
            result = solve(
                problem,
                algo=AlgoName.INCREMENTAL3,
                options=AlgoOptions(existential_strategy=strategy),
            )
            assert result == expected, (sentence, strategy)


def test_incremental3_rejects_unsupported_count_comparator():
    from wfomc.algo.incremental3.counting_state import (
        build_counting_state_for_normal_form,
    )
    from wfomc.fol.normal_form import normalize
    from wfomc.parser import parse_formula

    normal_form = normalize(parse_formula(r"\exists_{>2} X: U(X)"))

    with pytest.raises(ValueError, match="global count comparator"):
        build_counting_state_for_normal_form(normal_form)


def test_incremental3_rejects_embedded_count_definition():
    from wfomc.algo.incremental3.counting_state import (
        build_counting_state_for_normal_form,
    )
    from wfomc.fol.normal_form import normalize
    from wfomc.parser import parse_formula

    normal_form = normalize(
        parse_formula(r"\forall X: (P(X) | (\exists_=2 Y: R(X,Y)))")
    )

    with pytest.raises(UnsupportedFeatureError, match="embedded"):
        build_counting_state_for_normal_form(normal_form)
