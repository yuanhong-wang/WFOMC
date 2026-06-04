from pathlib import Path

from wfomc import Algo, UnaryEvidenceStrategy, parse_input, wfomc


ROOT = Path(__file__).parents[2]


def test_books_arrangement_linear_order_unary_evidence():
    model = ROOT / "models" / "linear_order_unary_evidence" / "books-arragement.wfomcs"
    problem = parse_input(str(model))

    auto_results = {
        "incremental": wfomc(problem, Algo.INCREMENTAL),
        "incremental3": wfomc(problem, Algo.INCREMENTAL3),
        "recursive": wfomc(problem, Algo.RECURSIVE),
    }
    ccs_results = {
        algo.value: wfomc(problem, algo, UnaryEvidenceStrategy.CCS)
        for algo in (Algo.INCREMENTAL, Algo.INCREMENTAL3, Algo.RECURSIVE)
    }

    assert auto_results == {name: 4 for name in auto_results}
    assert ccs_results == {name: 4 for name in ccs_results}


def test_incremental3_relative_config_basis_without_evidence():
    model = ROOT / "models" / "linear_order" / "head-middle-tail.wfomcs"
    problem = parse_input(str(model))

    incremental = wfomc(problem, Algo.INCREMENTAL)
    incremental3 = wfomc(problem, Algo.INCREMENTAL3)
    recursive = wfomc(problem, Algo.RECURSIVE)

    assert incremental3 == incremental == recursive
