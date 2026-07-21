"""Linear-order regressions, including interaction with unary evidence."""

import os
from pathlib import Path

from wfomc import AlgoName, AlgoOptions, EvidenceStrategy, parse_problem_file, solve


ROOT = Path(__file__).parents[2]
RUN_SLOW = os.environ.get("WFOMC_RUN_SLOW", "0") == "1"


def test_books_arrangement_linear_order_unary_evidence():
    model = (
        ROOT
        / "models"
        / "linear_order"
        / "unary_evidence"
        / "books-arragement.wfomcs"
    )
    problem = parse_problem_file(str(model))
    algorithms = (AlgoName.INCREMENTAL3,)
    if RUN_SLOW:
        algorithms = (
            AlgoName.INCREMENTAL,
            AlgoName.INCREMENTAL3,
            AlgoName.RECURSIVE,
        )

    auto_results = {algo: solve(problem, algo=algo) for algo in algorithms}
    ccs_results = {
        algo: solve(
            problem,
            algo=algo,
            options=AlgoOptions(evidence_strategy=EvidenceStrategy.CCS),
        )
        for algo in algorithms
    }

    assert all(result == 4 for result in auto_results.values()), auto_results
    assert all(result == 4 for result in ccs_results.values()), ccs_results


def test_incremental3_relative_config_basis_without_evidence():
    problem = parse_problem_file(
        str(ROOT / "models" / "linear_order" / "head-middle-tail.wfomcs")
    )

    results = {
        algo: solve(problem, algo=algo)
        for algo in (
            AlgoName.INCREMENTAL,
            AlgoName.INCREMENTAL3,
            AlgoName.RECURSIVE,
        )
    }

    first = next(iter(results.values()))
    assert all(result == first for result in results.values()), results
    assert first == 360
