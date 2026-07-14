from __future__ import annotations

import logging

from flint import fmpq

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.cell_graph import CellGraphData, build_cell_graphs
from wfomc.fol import FOLContext, conjunction


def _graph(*, with_predecessor: bool = False):
    ctx = FOLContext()
    x = ctx.variable("X")
    y = ctx.variable("Y")
    p = ctx.predicate("P", 1)
    relation = ctx.predicate("R", 2)
    formula = conjunction(p(x) | ~p(x), relation(x, y) | ~relation(x, y))
    predecessor = ctx.predicate("PRED", 2)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    data, graph_weight = next(
        build_cell_graphs(
            formula,
            {p: (fmpq(1), fmpq(1)), relation: (fmpq(1), fmpq(1))},
            arithmetic,
            leq_pred=ctx.predicate("LEQ", 2) if with_predecessor else None,
            predecessor_preds={1: predecessor} if with_predecessor else None,
        )
    )
    return data, relation, graph_weight, arithmetic


def test_builder_returns_plain_data_and_branch_weight():
    data, _relation, graph_weight, arithmetic = _graph()

    assert isinstance(data, CellGraphData)
    assert data.arithmetic is arithmetic
    assert len(data.cell_weights) == len(data.cells)
    assert len(data.pair_factors) == len(data.cells)
    assert graph_weight == arithmetic.one()


def test_pair_factor_stores_only_aggregated_weight():
    data, _predicate, _graph_weight, _arithmetic = _graph()

    factor = data.pair_factors[0][1]
    assert factor.total_weight > fmpq(0)
    assert factor.counting_weights == ()


def test_pair_weights_projects_all_unconditional_table_weights():
    data, _predicate, _graph_weight, _arithmetic = _graph()

    assert data.pair_weights() == tuple(
        tuple(factor.total_weight for factor in row) for row in data.pair_factors
    )


def test_predecessor_tables_are_keyed_by_order():
    data, _relation, _graph_weight, _arithmetic = _graph(with_predecessor=True)
    tables = dict(data.predecessor_pair_factors)

    assert set(tables) == {1}
    assert len(tables[1]) == len(data.cells)
    assert all(len(row) == len(data.cells) for row in tables[1])


def test_build_logs_pair_factor_runtime_by_phase(caplog):
    with caplog.at_level(logging.INFO, logger="wfomc.cell_graph"):
        _graph(with_predecessor=True)

    messages = tuple(record.getMessage() for record in caplog.records)
    assert any("phase=base" in message for message in messages)
    assert any("phase=predecessor[1]" in message for message in messages)
    assert any(
        "compile_ms=" in message and "evaluate_ms=" in message
        for message in messages
    )


def test_each_build_uses_the_supplied_arithmetic_context():
    first = ArithmeticContext(ArithmeticBackend.FMPQ)
    second = ArithmeticContext(ArithmeticBackend.FMPQ)
    ctx = FOLContext()
    predicate = ctx.predicate("ContextP", 1)
    formula = predicate(ctx.variable("X")) | ~predicate(ctx.variable("X"))

    first_data, _ = next(build_cell_graphs(formula, {}, first))
    second_data, _ = next(build_cell_graphs(formula, {}, second))

    assert first_data.arithmetic is first
    assert second_data.arithmetic is second
    assert first_data.pair_factors[0][0].total_weight == first.one()
    assert second_data.pair_factors[0][0].total_weight == second.one()
