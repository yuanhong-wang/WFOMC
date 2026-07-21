from __future__ import annotations

import pytest

from wfomc import Problem, ProblemInstance
from wfomc.fol import predicates

from benchmarks.cases import benchmark_case, benchmark_cases, benchmark_suite_names


def test_benchmark_suite_sizes_preserve_source_grids():
    assert len(benchmark_cases("core-smoke")) == 4
    assert len(benchmark_cases("core-main")) == 16
    assert len(benchmark_cases("core-exhaustive")) == 49
    assert len(benchmark_cases("c2")) == 47
    assert len(benchmark_cases("cardinality")) == 11
    assert len(benchmark_cases("unary")) == 54
    assert len(benchmark_cases("all")) == 165


def test_benchmark_case_keys_are_unique_and_deterministic():
    first = benchmark_cases("all")
    second = benchmark_cases("all")
    keys = tuple(case.key for case in first)

    assert keys == tuple(case.key for case in second)
    assert len(keys) == len(set(keys))
    assert benchmark_suite_names() == (
        "core-smoke",
        "core-main",
        "core-exhaustive",
        "c2",
        "cardinality",
        "unary",
        "all",
    )


def test_catalog_uses_only_semantic_family_and_encoding_names():
    actual = {
        category: {
            (case.family, case.variant)
            for case in benchmark_cases("all")
            if case.category == category
        }
        for category in ("core", "c2", "cardinality", "unary-cardinality")
    }

    assert actual == {
        "core": {
            (f"{k}-edge-disjoint-edge-covers", "default")
            for k in (2, 3, 4)
        } | {
            (f"{k}-neighbour-surjection-kernel", "default")
            for k in (2, 3, 4)
        } | {
            (f"properly-{k}-coloured-graph", "default")
            for k in (2, 3, 4, 5)
        } | {
            ("bi-total-relation", "default"),
            ("left-total-relation", "default"),
            ("loopless-bi-total-relation", "default"),
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
    for case in benchmark_cases("all"):
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


def test_representative_case_metadata_is_preserved():
    assert benchmark_case("core/bi-total-relation/n8").domain_size == 8
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


def test_core_main_and_exhaustive_membership_matches_source_catalog():
    main_keys = {case.key for case in benchmark_cases("core-main")}
    exhaustive_keys = {case.key for case in benchmark_cases("core-exhaustive")}

    assert main_keys < exhaustive_keys
    assert "core/3-edge-disjoint-edge-covers/n40" in main_keys
    assert "core/4-edge-disjoint-edge-covers/n16" in exhaustive_keys
    assert "core/loopless-digraph-without-isolates/n225" in exhaustive_keys


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
        "core/derangements/n8",
        "core/3-matchings/n10",
        "c2/3-regular/n10",
        "unary/row-column-Sx/exact/n20",
    ):
        with pytest.raises(KeyError, match="unknown benchmark case"):
            benchmark_case(key)
