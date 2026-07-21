from __future__ import annotations

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
    assert benchmark_case("core/row-column/n8").domain_size == 8
    assert benchmark_case("c2/3-regular-hand/n100").comparison_group == "c2-vs-hand-3-regular/n100"
    assert benchmark_case("cardinality/directed-3-regular/n15").correction_divisor == 36**15

    unary = benchmark_case("unary-structure/row-column-Sx/exact/n100")
    assert unary.purposes == frozenset(("structure", "clique-gate"))
    assert unary.variant == "exact"


def test_core_main_and_exhaustive_membership_matches_source_catalog():
    main_keys = {case.key for case in benchmark_cases("core-main")}
    exhaustive_keys = {case.key for case in benchmark_cases("core-exhaustive")}

    assert main_keys < exhaustive_keys
    assert "core/3-matchings/n40" in main_keys
    assert "core/4-matchings/n16" in exhaustive_keys
    assert "core/no-isolated-digraph/n225" in exhaustive_keys


def test_c2_direct_and_hand_cases_share_comparison_groups():
    for n in range(10, 101, 10):
        group = f"c2-vs-hand-3-regular/n{n}"
        direct = benchmark_case(f"c2/3-regular/n{n}")
        hand = benchmark_case(f"c2/3-regular-hand/n{n}")

        assert direct.comparison_group == group
        assert hand.comparison_group == group
