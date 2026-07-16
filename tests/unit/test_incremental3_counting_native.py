from __future__ import annotations

import pytest

from wfomc.algo import AlgoName, AlgoOptions, ExistentialStrategy
from wfomc.algo.incremental3.counting_kernel import (
    ConfigSpace,
    _build_elimination_orders,
)
from wfomc.algo.incremental3.counting_state import CountingState, RowCounterSpec
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


def test_elimination_order_is_fail_first_and_independent_of_state_enumeration():
    unconstrained = (0,)
    constrained = (1,)
    branching = (2,)
    states = (unconstrained, constrained, branching)
    deterministic = {((0,), (0,)): 1}
    two_outcomes = {
        ((0,), (0,)): 1,
        ((1,), (1,)): 1,
    }
    transitions = {
        (unconstrained, unconstrained): deterministic,
        (unconstrained, constrained): deterministic,
        (unconstrained, branching): deterministic,
        (constrained, unconstrained): deterministic,
        (constrained, constrained): deterministic,
        # constrained -> branching is deliberately incompatible.
        (branching, unconstrained): two_outcomes,
        (branching, constrained): two_outcomes,
        (branching, branching): two_outcomes,
    }

    target_order, other_orders = _build_elimination_orders(transitions, states)
    reversed_target_order, _ = _build_elimination_orders(
        transitions,
        tuple(reversed(states)),
    )

    assert target_order[0] == constrained
    assert reversed_target_order[0] == constrained
    assert other_orders[constrained][0] == branching


def test_positive_mod_cell_wraps_initial_remainder():
    predicate = Predicate("Remainder", 2)
    state = CountingState(
        (
            RowCounterSpec(
                predicate=predicate,
                true_transitions=(1, 0),
                accepting_states=frozenset({0}),
            ),
        )
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

    assert len(state.row_counters) == 1
    assert state.row_counters[0].predicate.name == "E"
    assert state.row_counters[0].true_transitions == (1, 0)
    assert state.row_counters[0].accepting_states == frozenset({1})
    assert masks.constraints == []


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
    assert [
        (spec.predicate.name, spec.comparator, spec.count)
        for spec in masks.constraints
    ] == [
        ("U", "=", 1)
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


@pytest.mark.parametrize(
    "algo",
    (
        AlgoName.STANDARD,
        AlgoName.FAST,
        AlgoName.FASTV2,
        AlgoName.INCREMENTAL,
        AlgoName.RECURSIVE,
        AlgoName.PROPOSITIONAL,
        AlgoName.PROPOSITIONAL_REDUCED,
    ),
)
@pytest.mark.parametrize(
    "strategy",
    (ExistentialStrategy.COUNTING, ExistentialStrategy.SKOLEM),
)
def test_existential_strategy_is_rejected_outside_incremental3(
    algo: AlgoName,
    strategy: ExistentialStrategy,
):
    problem = parse_problem(
        r"""
\forall X: (\exists Y: (R(X,Y)))
domain = 2
"""
    )

    with pytest.raises(
        UnsupportedFeatureError,
        match="only configurable for incremental3",
    ):
        compile_problem(
            problem,
            algo=algo,
            options=AlgoOptions(existential_strategy=strategy),
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


def test_incremental3_materializes_strict_count_comparator():
    from wfomc.algo.incremental3.counting_state import (
        build_counting_state_for_normal_form,
    )
    from wfomc.fol.normal_form import normalize
    from wfomc.parser import parse_formula

    normal_form = normalize(parse_formula(r"\exists_{>2} X: U(X)"))

    _state, masks = build_counting_state_for_normal_form(normal_form)

    assert masks.constraints[0].comparator == ">"
    assert masks.constraints[0].count == 2


def test_incremental3_materializes_embedded_count_definition():
    from wfomc.algo.incremental3.counting_state import (
        build_counting_state_for_normal_form,
    )
    from wfomc.fol.normal_form import normalize
    from wfomc.parser import parse_formula

    normal_form = normalize(
        parse_formula(r"\forall X: (P(X) | (\exists_=2 Y: R(X,Y)))")
    )

    state, _masks = build_counting_state_for_normal_form(normal_form)

    assert len(state.row_counters) == 1
    assert state.row_counters[0].marker_predicate is not None
    assert state.row_counters[0].marker_predicate.arity == 1
