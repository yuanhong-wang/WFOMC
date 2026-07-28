from __future__ import annotations

from math import factorial
from pathlib import Path

import pytest

from benchmarks.cases import benchmark_case as global_benchmark_case
from benchmarks.run_paper import (
    PAPER_BENCHMARK_CASES,
    PAPER_DOMAIN_GRIDS,
    _friends_smokers_problem,
    _typed_sparse_problem,
    main,
    paper_benchmark_case,
    paper_benchmark_cases,
)
from wfomc import AlgoName, solve


EXPECTED_DOMAINS = {
    "permutations": (20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 125),
    "undirected-2-regular": (20, 40, 60, 80, 100, 120, 140, 160, 180, 200),
    "undirected-3-regular": (20, 30, 40, 50, 60, 70, 80, 90, 100),
    "undirected-4-regular": (
        12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50, 52, 56, 60,
    ),
    "properly-2-coloured-graph": (
        100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000,
    ),
    "properly-3-coloured-graph": (
        50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300,
    ),
    "properly-4-coloured-graph": (
        20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150,
    ),
    "properly-5-coloured-graph": (
        16, 24, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
    ),
    "derangements": (50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300),
    "endofunctions": (40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240),
    "loopless-digraph-without-isolates": (
        50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600,
    ),
    "2-edge-disjoint-perfect-matchings": (
        20, 40, 60, 80, 100, 150, 200, 250, 300, 400,
    ),
    "3-edge-disjoint-perfect-matchings": (
        10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    ),
    "4-edge-disjoint-perfect-matchings": (6, 8, 10, 12, 14, 16),
    "typed-path-relation-k8": (
        20, 40, 60, 80, 100, 120, 160, 200, 250, 300, 400, 500,
    ),
    "typed-tree-relation-k8": (
        20, 40, 60, 80, 100, 120, 160, 200, 250, 300, 400, 500,
    ),
    "typed-cycle-relation-k8": (
        20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200,
    ),
    "typed-asymmetric-relation-k8": (
        20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 300,
    ),
    "properly-3-coloured-undirected-3-regular": (
        6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    ),
    "properly-4-coloured-undirected-3-regular": (
        4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    ),
    "properly-5-coloured-undirected-3-regular": (
        4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    ),
    "properly-4-coloured-undirected-2-regular": (
        4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    ),
    "properly-4-coloured-undirected-4-regular": (
        6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    ),
    "friends-smokers": (
        20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
    ),
    "academic-advising": (
        4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    ),
    "id2-gene-regulation": (
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
    ),
    "imdb-worked-under-fo2": (
        2, 4, 6, 8, 12, 16, 24, 32, 40, 48,
        56, 64, 72, 80, 96, 112, 128, 144, 160,
    ),
    "webkb-link-classification": (
        4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    ),
}


def test_paper_catalog_uses_the_fixed_domain_grids() -> None:
    assert paper_benchmark_cases() is PAPER_BENCHMARK_CASES
    assert len(PAPER_DOMAIN_GRIDS) == 28
    assert len(PAPER_BENCHMARK_CASES) == 336
    assert all(case.category == "core" for case in PAPER_BENCHMARK_CASES)
    actual = {
        family: tuple(
            case.domain_size
            for case in PAPER_BENCHMARK_CASES
            if case.family == family
        )
        for family in EXPECTED_DOMAINS
    }
    assert actual == EXPECTED_DOMAINS
    assert len({case.key for case in PAPER_BENCHMARK_CASES}) == 336


def test_paper_case_lookup_supports_fixed_grid_only() -> None:
    case = paper_benchmark_case(
        "core/properly-2-coloured-graph/n2000"
    )
    assert case.domain_size == 2000

    try:
        paper_benchmark_case("core/properly-2-coloured-graph/n2001")
    except KeyError as error:
        assert "unknown Core paper case" in str(error)
    else:
        raise AssertionError("out-of-grid paper case should not be found")

    typed = paper_benchmark_case("core/typed-path-relation-k8/n500")
    assert typed.domain_size == 500
    assert typed.purposes == frozenset(
        ("paper", "typed-sparse-interaction")
    )


@pytest.mark.parametrize(
    "key",
    (
        "core/typed-path-relation-k8/n20",
        (
            "core/properly-4-coloured-undirected-3-regular/"
            "fo2-cardinality-reduction/n4"
        ),
        (
            "core/properly-4-coloured-undirected-4-regular/"
            "fo2-cardinality-reduction/n6"
        ),
        "core/friends-smokers/n20",
        "core/academic-advising/n4",
        "core/id2-gene-regulation/n2",
        "core/imdb-worked-under-fo2/n2",
        "core/webkb-link-classification/n4",
    ),
)
def test_extra_cases_remain_paper_local(key: str) -> None:
    with pytest.raises(KeyError, match="unknown benchmark case"):
        global_benchmark_case(key)


def test_typed_sparse_sentence_is_domain_independent_and_algorithms_agree() -> None:
    small = _typed_sparse_problem("path", 4)
    larger = _typed_sparse_problem("path", 5)

    assert small.problem.cache_key_parts() == larger.problem.cache_key_parts()
    results = [
        solve(small, algo=algorithm)
        for algorithm in (
            AlgoName.BOUNDARY_PROFILE,
            AlgoName.FAST,
            AlgoName.INCREMENTAL3,
        )
    ]
    assert results[1:] == results[:-1]


@pytest.mark.parametrize(
    ("colour_count", "degree", "domain_size"),
    (
        (3, 3, 6),
        (4, 3, 4),
        (5, 3, 4),
        (4, 2, 4),
        (4, 4, 6),
    ),
)
def test_coloured_regular_selects_original_c2_and_counts_agree(
    colour_count: int,
    degree: int,
    domain_size: int,
) -> None:
    case = paper_benchmark_case(
        f"core/properly-{colour_count}-coloured-undirected-{degree}-regular/"
        f"fo2-cardinality-reduction/n{domain_size}"
    )

    assert case.input_variant_for("boundary-profile") == (
        "fo2-cardinality-reduction"
    )
    assert case.input_variant_for("fast") == "fo2-cardinality-reduction"
    assert case.input_variant_for("incremental3") == "original-c2"
    assert case.correction_divisor_for("boundary-profile") == (
        factorial(degree) ** domain_size
    )
    assert case.correction_divisor_for("incremental3") == 1

    reduced = solve(
        case.build_problem_for("boundary-profile"),
        algo=AlgoName.BOUNDARY_PROFILE,
    )
    fast = solve(
        case.build_problem_for("fast"),
        algo=AlgoName.FAST,
    )
    original = solve(
        case.build_problem_for("incremental3"),
        algo=AlgoName.INCREMENTAL3,
    )
    assert fast == reduced
    assert int(reduced) // case.correction_divisor_for("boundary-profile") == int(
        original
    )


def test_new_model_problems_are_domain_independent_and_algorithms_agree() -> None:
    coloured_small = paper_benchmark_case(
        "core/properly-4-coloured-undirected-3-regular/"
        "fo2-cardinality-reduction/n4"
    )
    coloured_larger = paper_benchmark_case(
        "core/properly-4-coloured-undirected-3-regular/"
        "fo2-cardinality-reduction/n6"
    )
    friends_small = _friends_smokers_problem(4)
    friends_larger = _friends_smokers_problem(5)

    reduced_small = coloured_small.build_problem().problem
    reduced_larger = coloured_larger.build_problem().problem
    assert repr(reduced_small.sentence) == repr(reduced_larger.sentence)
    assert reduced_small.weights == reduced_larger.weights
    assert (
        reduced_small.cardinality_constraints.constraints[0].rhs,
        reduced_larger.cardinality_constraints.constraints[0].rhs,
    ) == (12, 18)
    assert (
        coloured_small.build_problem_for("incremental3").problem.cache_key_parts()
        == coloured_larger.build_problem_for(
            "incremental3"
        ).problem.cache_key_parts()
    )
    assert (
        friends_small.problem.cache_key_parts()
        == friends_larger.problem.cache_key_parts()
    )
    results = [
        solve(friends_small, algo=algorithm)
        for algorithm in (
            AlgoName.BOUNDARY_PROFILE,
            AlgoName.FAST,
            AlgoName.INCREMENTAL3,
        )
    ]
    assert results[1:] == results[:-1]


@pytest.mark.parametrize(
    "family",
    (
        "academic-advising",
        "id2-gene-regulation",
        "imdb-worked-under-fo2",
        "webkb-link-classification",
    ),
)
def test_relational_mln_models_are_domain_independent_and_algorithms_agree(
    family: str,
) -> None:
    small = paper_benchmark_case(f"core/{family}/n4").build_problem()
    larger = paper_benchmark_case(f"core/{family}/n6").build_problem()

    assert small.problem.cache_key_parts() == larger.problem.cache_key_parts()
    results = [
        solve(small, algo=algorithm)
        for algorithm in (
            AlgoName.BOUNDARY_PROFILE,
            AlgoName.FAST,
            AlgoName.INCREMENTAL3,
        )
    ]
    assert results[1:] == results[:-1]


def test_paper_runner_uses_formal_experiment_defaults(monkeypatch) -> None:
    captured = {}

    def fake_run(cases, args, *, worker_module):
        captured["cases"] = cases
        captured["out"] = args.out
        captured["protocol"] = args.protocol
        captured["timeout"] = args.timeout
        captured["repetitions"] = args.repetitions
        captured["worker_module"] = worker_module
        return 0

    monkeypatch.setattr("benchmarks.run_paper.run_catalog", fake_run)

    assert main([]) == 0
    assert captured["cases"] is PAPER_BENCHMARK_CASES
    assert captured["out"] == Path("benchmark-results/paper-core").resolve()
    assert captured["protocol"] == "cold-stop"
    assert captured["timeout"] == 300.0
    assert captured["repetitions"] == 3
    assert captured["worker_module"] == "benchmarks.run_paper"
