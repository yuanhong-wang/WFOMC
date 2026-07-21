from __future__ import annotations

import pytest

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.algo.treewidth.input import (
    Factor,
    FactorGraph,
    FactorGraphKind,
    TreeBag,
    TreeDecomposition,
    TreeDecompositionInput,
)
from wfomc.algo import AlgoName
from wfomc.errors import UnsupportedFeatureError
from wfomc.engine import solve as engine_solve
from wfomc.algo.treewidth.solve import solve
from wfomc.parser import parse_problem_file


def _arithmetic() -> ArithmeticContext:
    return ArithmeticContext(ArithmeticBackend.FMPQ)


def test_tree_decomposition_input_carries_factor_graph_contract():
    factor_graph = FactorGraph(
        kind=FactorGraphKind.LIFTED_CELL,
        variables=("cell-count", "binary-table"),
        factors=(
            Factor(
                name="cell-pair",
                scope=("cell-count", "binary-table"),
                weight="phi",
            ),
        ),
    )
    decomposition = TreeDecomposition(
        bags=(
            TreeBag(
                name="root",
                variables=frozenset({"cell-count", "binary-table"}),
            ),
        ),
        root="root",
    )
    decomposition.validate_against(factor_graph)

    algo_input = TreeDecompositionInput(
        arithmetic=_arithmetic(),
        factor_graph=factor_graph,
        bags=decomposition.bags,
        local_factors=factor_graph.factors,
    )

    assert algo_input.factor_graph.kind == FactorGraphKind.LIFTED_CELL
    assert decomposition.width == 1
    assert algo_input.local_factors[0].name == "cell-pair"


def test_treewidth_contract_supports_ground_evidence_factors():
    factor_graph = FactorGraph(
        kind=FactorGraphKind.GROUND,
        variables=("P(a)", "R(a,b)"),
        factors=(Factor(name="clause-0", scope=("P(a)", "R(a,b)")),),
        evidence_factors=(Factor(name="evidence-P-a", scope=("P(a)",), weight=0),),
    )
    decomposition = TreeDecomposition(
        bags=(TreeBag(name="root", variables=frozenset({"P(a)", "R(a,b)"})),),
    )

    decomposition.validate_against(factor_graph)

    assert factor_graph.variable_set == frozenset({"P(a)", "R(a,b)"})
    assert factor_graph.all_factors[-1].name == "evidence-P-a"


def test_tree_decomposition_rejects_uncovered_factor_scope():
    factor_graph = FactorGraph(
        kind=FactorGraphKind.GROUND,
        factors=(Factor(name="binary-factor", scope=("x", "y")),),
    )
    decomposition = TreeDecomposition(
        bags=(TreeBag(name="root", variables=frozenset({"x"})),),
    )

    with pytest.raises(ValueError, match="not covered"):
        decomposition.validate_against(factor_graph)


def test_bounded_treewidth_algo_rejects_until_solver_is_installed():
    algo = solve

    with pytest.raises(UnsupportedFeatureError, match="bounded-treewidth"):
        algo(
            TreeDecompositionInput(
                arithmetic=_arithmetic(),
            )
        )


def test_compile_problem_treewidth_reduction_raises_unsupported():
    problem = parse_problem_file("models/2-colored-graph.wfomcs")

    with pytest.raises(UnsupportedFeatureError, match="bounded-treewidth"):
        engine_solve(problem, algo=AlgoName.BOUNDED_TREEWIDTH)
