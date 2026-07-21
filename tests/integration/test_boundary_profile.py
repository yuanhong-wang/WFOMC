from __future__ import annotations

import pytest
from flint import arb

from wfomc import (
    AlgoName,
    AlgoOptions,
    BoundaryProfileOptions,
    EvidenceStrategy,
    parse_problem_file,
    solve,
)
from wfomc.weights import WeightOptions


@pytest.mark.parametrize(
    "model_path",
    (
        "models/2-colored-graph.wfomcs",
        "models/friends-smokes.wfomcs",
        "models/regular_graphs/3-regular-2-colored-graph.wfomcs",
    ),
)
def test_boundary_profile_matches_fastv2(model_path):
    problem = parse_problem_file(model_path)

    assert solve(problem, algo=AlgoName.BOUNDARY_PROFILE) == solve(
        problem,
        algo=AlgoName.FASTV2,
    )


def test_boundary_profile_supports_cardinality_mpoly_decoder():
    problem = parse_problem_file("models/cardinality_constraints_example.wfomcs")

    assert solve(problem, algo=AlgoName.BOUNDARY_PROFILE) == solve(
        problem,
        algo=AlgoName.FASTV2,
    )


def test_boundary_profile_supports_explicit_univariate_backend():
    problem = parse_problem_file("models/cardinality_constraints_example.wfomcs")
    options = AlgoOptions(
        weight_options=WeightOptions(exact_symbolic_backend="fmpq_poly")
    )

    assert solve(
        problem,
        algo=AlgoName.BOUNDARY_PROFILE,
        options=options,
    ) == solve(problem, algo=AlgoName.FASTV2, options=options)


def test_boundary_profile_supports_ccs_unary_evidence():
    problem = parse_problem_file("models/unary_evidence/evidence-only.wfomcs")
    options = AlgoOptions(evidence_strategy=EvidenceStrategy.CCS)

    assert solve(
        problem,
        algo=AlgoName.BOUNDARY_PROFILE,
        options=options,
    ) == solve(problem, algo=AlgoName.FASTV2, options=options)


def test_boundary_profile_reference_domain_planning_matches_fastv2():
    problem = parse_problem_file("models/friends-smokes.wfomcs")
    options = AlgoOptions(
        boundary_profile_options=BoundaryProfileOptions(
            tree_reference_domain_size=8,
        )
    )

    assert solve(
        problem,
        algo=AlgoName.BOUNDARY_PROFILE,
        options=options,
    ) == solve(problem, algo=AlgoName.FASTV2)


@pytest.mark.parametrize(
    ("backend", "expected_type"),
    (("float", float), ("arb", arb)),
)
def test_boundary_profile_supports_rounded_scalar_backends(
    backend,
    expected_type,
):
    problem = parse_problem_file("models/2-colored-graph.wfomcs")
    options = AlgoOptions(
        weight_options=WeightOptions(
            precision="round",
            rounded_backend=backend,
        )
    )

    result = solve(
        problem,
        algo=AlgoName.BOUNDARY_PROFILE,
        options=options,
    )

    assert isinstance(result.raw, expected_type)
    assert float(result) == pytest.approx(330626.0)
