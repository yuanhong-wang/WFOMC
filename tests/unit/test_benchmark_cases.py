from __future__ import annotations

import pytest

import benchmarks.cases as catalog
from wfomc import (
    AlgoName,
    Problem,
    ProblemInstance,
    compile_problem,
    instantiate_problem,
    solve,
)
from wfomc.arithmetic import ArithmeticBackend
from wfomc.engine.features import analyze_problem_features
from wfomc.fol import FormulaKind, predicates, walk

from benchmarks.cases import BENCHMARK_CASES, benchmark_case, benchmark_cases


def test_flat_benchmark_catalog_preserves_every_source_case():
    cases = benchmark_cases()

    assert cases is BENCHMARK_CASES
    assert len(cases) == 165
    assert {
        category: sum(case.category == category for case in cases)
        for category in ("core", "c2", "cardinality", "unary-cardinality")
    } == {
        "core": 53,
        "c2": 47,
        "cardinality": 11,
        "unary-cardinality": 54,
    }
    with pytest.raises(TypeError):
        benchmark_cases("all")


def test_benchmark_case_keys_are_unique_and_deterministic():
    first = benchmark_cases()
    second = benchmark_cases()
    keys = tuple(case.key for case in first)

    assert keys == tuple(case.key for case in second)
    assert len(keys) == len(set(keys))


def test_catalog_uses_only_semantic_family_and_encoding_names():
    actual = {
        category: {
            (case.family, case.variant)
            for case in benchmark_cases()
            if case.category == category
        }
        for category in ("core", "c2", "cardinality", "unary-cardinality")
    }

    assert actual == {
        "core": {
            (f"{k}-edge-disjoint-perfect-matchings", "fo2-cardinality-reduction")
            for k in (2, 3, 4)
        } | {
            (f"undirected-{k}-regular", "fo2-cardinality-reduction")
            for k in (2, 3, 4)
        } | {
            (f"properly-{k}-coloured-graph", "default")
            for k in (2, 3, 4, 5)
        } | {
            ("permutations", "fo2-cardinality-reduction"),
            ("endofunctions", "fo2-cardinality-reduction"),
            ("derangements", "fo2-cardinality-reduction"),
            ("loopless-digraph-without-isolates", "default"),
        },
        "c2": {
            ("undirected-3-regular", "direct-c2"),
            ("undirected-3-regular", "fo2-cardinality-reduction"),
            ("properly-3-coloured-undirected-3-regular", "direct-c2"),
            ("directed-3-in-3-out-regular", "direct-c2"),
        },
        "cardinality": {
            (
                "properly-4-coloured-undirected-3-regular",
                "fo2-cardinality-reduction",
            ),
            ("directed-3-in-3-out-regular", "fo2-cardinality-reduction"),
        },
        "unary-cardinality": {
            (family, variant)
            for family in (
                "bi-total-relation/sx-cardinality",
                "left-total-relation/s-cardinality",
                "loopless-digraph-without-isolates/s-cardinality",
                "properly-4-coloured-graph/c1-cardinality",
            )
            for variant in ("unconstrained", "exact", "interval")
        },
    }


def test_every_benchmark_case_builds_a_current_typed_problem():
    for case in benchmark_cases():
        problem = case.build_problem()
        declared = predicates(problem.problem.sentence)

        assert isinstance(problem, ProblemInstance), case.key
        assert isinstance(problem.problem, Problem), case.key
        assert len(problem.domain) == case.domain_size, case.key
        assert case.correction_divisor > 0, case.key
        assert set(problem.problem.weights) <= declared, case.key
        assert all(
            term.predicate in declared
            for constraint in problem.problem.cardinality_constraints.constraints
            for term in constraint.terms
        ), case.key


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        (catalog._core_permutation_case(3), 6),
        (catalog._core_derangement_case(3), 2),
        (catalog._core_endofunction_case(3), 27),
        (catalog._core_regular_case(2, 4), 3),
        (catalog._core_matching_case(2, 4), 6),
        (catalog._core_matching_case(3, 4), 6),
        (catalog._core_loopless_no_isolates_case(2), 3),
    ),
)
def test_cardinality_complete_core_cases_have_expected_small_counts(
    case,
    expected,
):
    raw = solve(case.build_problem(), algo=AlgoName.INCREMENTAL3).raw

    assert raw == expected * case.correction_divisor


@pytest.mark.parametrize(
    "case",
    (
        catalog._core_permutation_case(3),
        catalog._core_regular_case(2, 4),
        catalog._core_regular_case(3, 4),
        catalog._core_regular_case(4, 5),
        catalog._core_derangement_case(3),
        catalog._core_endofunction_case(3),
        catalog._core_loopless_no_isolates_case(3),
        catalog._core_matching_case(2, 4),
        catalog._core_matching_case(3, 4),
        catalog._core_matching_case(4, 4),
    ),
)
def test_incremental3_uses_original_c2_for_reduced_core_families(case):
    reduced = case.build_problem()
    direct = case.build_problem_for("incremental3")

    assert (
        reduced.problem.cardinality_constraints.constraints
        or any(
            -1 in weights
            for weights in reduced.problem.weights.values()
        )
    )
    assert not direct.problem.cardinality_constraints.constraints
    assert (
        analyze_problem_features(direct.problem).has_c2_counting
        or any(
            node.op is FormulaKind.EXISTS
            for node in walk(direct.problem.sentence)
        )
    )
    assert all(
        -1 not in weights
        for weights in direct.problem.weights.values()
    )
    assert case.input_variant_for("incremental3") == "original-c2"
    assert case.correction_divisor_for("incremental3") == 1
    assert direct.problem.cache_key_parts() != reduced.problem.cache_key_parts()


@pytest.mark.parametrize(
    "case",
    (
        catalog._core_permutation_case(3),
        catalog._core_regular_case(2, 4),
        catalog._core_regular_case(3, 4),
        catalog._core_regular_case(4, 5),
        catalog._core_derangement_case(3),
        catalog._core_endofunction_case(3),
        catalog._core_loopless_no_isolates_case(3),
        catalog._core_matching_case(2, 4),
        catalog._core_matching_case(3, 4),
        catalog._core_matching_case(4, 4),
    ),
)
def test_original_c2_and_reduced_core_inputs_have_same_small_count(case):
    direct = solve(
        case.build_problem_for("incremental3"),
        algo=AlgoName.INCREMENTAL3,
    ).raw
    reduced = solve(case.build_problem(), algo=AlgoName.INCREMENTAL3).raw

    assert direct == reduced / case.correction_divisor


@pytest.mark.parametrize(("layers", "domain_size"), ((2, 4), (3, 6), (4, 8)))
def test_matching_cases_use_one_aggregate_cardinality_constraint(
    layers,
    domain_size,
):
    problem = catalog._core_matching_case(layers, domain_size).build_problem()
    constraints = problem.problem.cardinality_constraints.constraints

    assert len(constraints) == 1
    constraint = constraints[0]
    assert constraint.comparator.value == "="
    assert constraint.rhs == layers * domain_size
    assert {
        term.predicate.name: term.coefficient
        for term in constraint.terms
    } == {f"E{index}": 1 for index in range(1, layers + 1)}


def test_matching_aggregate_uses_one_truncated_series_marker():
    instance = catalog._core_matching_case(3, 20).build_problem()
    compiled = compile_problem(
        instance.problem,
        algo=AlgoName.BOUNDARY_PROFILE,
    )
    artifacts = instantiate_problem(compiled, instance.domain)
    arithmetic = artifacts.branches[0].algo_input.arithmetic

    assert arithmetic.backend is ArithmeticBackend.FMPQ_SERIES
    assert arithmetic.symbolic_variables == ("__wfomc_cardinality_0",)
    assert arithmetic.degree_limits == (("__wfomc_cardinality_0", 60),)


def test_representative_case_metadata_is_preserved():
    assert (
        benchmark_case("core/permutations/fo2-cardinality-reduction/n8").domain_size
        == 8
    )
    regular = benchmark_case(
        "core/undirected-3-regular/fo2-cardinality-reduction/n30"
    )
    assert regular.correction_divisor == 6**30
    assert regular.comparison_group == "undirected-3-regular/n30"
    reduced = benchmark_case(
        "c2/undirected-3-regular/fo2-cardinality-reduction/n100"
    )
    assert reduced.comparison_group == "undirected-3-regular/n100"
    directed = benchmark_case(
        "cardinality/directed-3-in-3-out-regular/"
        "fo2-cardinality-reduction/n15"
    )
    assert directed.correction_divisor == 36**15

    unary = benchmark_case(
        "unary/bi-total-relation/sx-cardinality/exact/n100"
    )
    assert unary.purposes == frozenset(("structure", "clique-gate"))
    assert unary.variant == "exact"


def test_flat_catalog_contains_representative_core_cases():
    keys = {case.key for case in benchmark_cases()}

    assert (
        "core/3-edge-disjoint-perfect-matchings/"
        "fo2-cardinality-reduction/n40"
    ) in keys
    assert (
        "core/4-edge-disjoint-perfect-matchings/"
        "fo2-cardinality-reduction/n16"
    ) in keys
    assert "core/loopless-digraph-without-isolates/n225" in keys


def test_c2_direct_and_reduced_cases_share_semantic_family_and_group():
    for n in range(10, 101, 10):
        group = f"undirected-3-regular/n{n}"
        direct = benchmark_case(f"c2/undirected-3-regular/direct-c2/n{n}")
        reduced = benchmark_case(
            f"c2/undirected-3-regular/fo2-cardinality-reduction/n{n}"
        )

        assert direct.comparison_group == group
        assert reduced.comparison_group == group
        assert direct.family == reduced.family == "undirected-3-regular"
        assert direct.variant == "direct-c2"
        assert reduced.variant == "fo2-cardinality-reduction"


def test_historical_case_keys_are_not_aliases():
    for key in (
        "core/row-column/n8",
        "core/bi-total-relation/n8",
        "core/loopless-bi-total-relation/n8",
        "core/left-total-relation/n80",
        "core/3-neighbour-surjection-kernel/n30",
        "core/3-edge-disjoint-edge-covers/n10",
        "core/3-matchings/n10",
        "c2/3-regular/n10",
        "unary/row-column-Sx/exact/n20",
    ):
        with pytest.raises(KeyError, match="unknown benchmark case"):
            benchmark_case(key)
