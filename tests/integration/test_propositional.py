"""Cross-check the GANAK-backed counter against lifted algorithms."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wfomc import AlgoName, parse_problem_file, solve
from wfomc.fol.grounding import (
    LinearOrderEncoding,
    resolve_linear_order_encoding,
)
from wfomc.ganak import GanakError, find_ganak
from wfomc.engine.features import analyze_problem_features


RUN_SLOW = os.environ.get("WFOMC_RUN_SLOW", "0") == "1"
AXIOMS_MODE = resolve_linear_order_encoding(None) is LinearOrderEncoding.AXIOMS
ROOT = Path(__file__).parents[2]
IN_SCOPE_DIRS = (
    ROOT / "models",
    ROOT / "models" / "unary_evidence",
    ROOT / "models" / "linear_order",
    ROOT / "models" / "linear_order" / "unary_evidence",
    ROOT / "models" / "linear_order" / "predk",
    ROOT / "models" / "counting_quantifiers",
)
CIRCULAR_MODELS = (
    ROOT / "models" / "MATH" / "8.wfomcs",
    ROOT / "models" / "MATH" / "33.wfomcs",
)
SLOW_UNDER_AXIOMS = frozenset(
    ("models/linear_order/predk/predecessor.wfomcs",)
)
SLOW_DIRECT_GROUNDING = frozenset(
    (
        "models/linear_order/linear_order_perm.wfomcs",
        "models/linear_order/predk/predecessor.wfomcs",
    )
)


def _collect_models() -> tuple[Path, ...]:
    paths = []
    for directory in IN_SCOPE_DIRS:
        paths.extend(sorted(directory.glob("*.wfomcs")))
        paths.extend(sorted(directory.glob("*.mln")))
    return tuple(paths)


def _ganak_available() -> bool:
    try:
        find_ganak()
    except GanakError:
        return False
    return True


@pytest.fixture(scope="session", autouse=True)
def _require_ganak():
    if not _ganak_available():
        pytest.skip("ganak binary not found; run `uv run wfomc-install-ganak`")


def _uses_multi_k_pred(problem) -> bool:
    return any(
        order > 1
        for order, _predicate in analyze_problem_features(
            problem.problem
        ).predecessor_predicates
    )


def _uses_general_cardinality(problem) -> bool:
    from collections import defaultdict

    from wfomc.cardinality_constraints import Comparator

    for constraint in problem.problem.cardinality_constraints.constraints:
        coefficients = defaultdict(int)
        for term in constraint.terms:
            coefficients[term.predicate] += term.coefficient
        nonzero = [value for value in coefficients.values() if value]
        if (
            nonzero != [1]
            or constraint.comparator
            not in {Comparator.LE, Comparator.EQ, Comparator.GE}
        ):
            return True
    return False


def _reference_algo(problem) -> AlgoName:
    features = analyze_problem_features(problem.problem)
    if features.has_predk or features.has_circular_pred:
        return AlgoName.INCREMENTAL
    if features.has_linear_order or features.has_c2_counting:
        return AlgoName.INCREMENTAL3
    return AlgoName.FASTV2


@pytest.mark.parametrize("model_file", _collect_models(), ids=lambda path: path.stem)
def test_propositional_matches_reference(model_file: Path):
    relative = str(model_file.relative_to(ROOT))
    if AXIOMS_MODE and relative in SLOW_UNDER_AXIOMS and not RUN_SLOW:
        pytest.skip(f"{relative} is slow under FO3 axiomatization")
    if relative in SLOW_DIRECT_GROUNDING and not RUN_SLOW:
        pytest.skip(f"{relative} is slow under direct grounding")
    problem = parse_problem_file(str(model_file))
    if _uses_multi_k_pred(problem):
        pytest.skip("PREDk for k > 1 is outside propositional scope")
    if _uses_general_cardinality(problem):
        pytest.skip("general linear cardinality is outside direct propositional scope")

    reference_algo = _reference_algo(problem)
    reference = solve(problem, algo=reference_algo)
    propositional = solve(problem, algo=AlgoName.PROPOSITIONAL)

    assert propositional == reference, (
        f"propositional={propositional} != {reference_algo.value}={reference} "
        f"for {model_file}"
    )


@pytest.mark.skipif(
    AXIOMS_MODE and not RUN_SLOW,
    reason="Circular axiomatization is slow; use pin mode or WFOMC_RUN_SLOW=1",
)
@pytest.mark.parametrize(
    "model_file",
    tuple(path for path in CIRCULAR_MODELS if path.exists()),
    ids=lambda path: path.stem,
)
def test_propositional_matches_reference_circular(model_file: Path):
    problem = parse_problem_file(str(model_file))

    reference = solve(problem, algo=AlgoName.INCREMENTAL)
    propositional = solve(problem, algo=AlgoName.PROPOSITIONAL)

    assert propositional == reference


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            r"""
\forall X: (P(X) | ~P(X))
domain = 2
""",
            4,
        ),
        (
            r"""
\forall X: (\exists Y: R(X,Y))
domain = 2
""",
            9,
        ),
        (
            r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 2
""",
            4,
        ),
        (
            r"""
\forall X: (P(X) | ~P(X))
domain = 3
|P| = 1
""",
            3,
        ),
    ),
    ids=("ordinary", "existential", "row-counting", "global-cardinality"),
)
def test_direct_and_reduced_propositional_agree(source: str, expected: int):
    from wfomc import parse_problem

    problem = parse_problem(source)

    direct = solve(problem, algo=AlgoName.PROPOSITIONAL)
    reduced = solve(problem, algo=AlgoName.PROPOSITIONAL_REDUCED)

    assert direct == expected
    assert reduced == expected
    assert direct == reduced


@pytest.mark.parametrize(
    ("comparator", "expected"),
    (("<=", 4), ("=", 3), (">=", 7)),
)
def test_direct_propositional_counts_simple_global_cardinality(
    comparator: str,
    expected: int,
):
    from wfomc import parse_problem

    problem = parse_problem(
        rf"""
\forall X: (P(X) | ~P(X))
domain = 3
|P| {comparator} 1
"""
    )

    assert solve(problem, algo=AlgoName.PROPOSITIONAL) == expected


def test_direct_propositional_materializes_constraint_only_binary_predicate():
    from wfomc import (
        CardinalityConstraints,
        CardinalityTerm,
        Comparator,
        Domain,
        LinearCardinalityConstraint,
        Problem,
    )
    from wfomc.fol import FOLContext

    fol = FOLContext()
    relation = fol.predicate("R", 2)
    domain = frozenset(fol.constant(name) for name in ("a", "b"))
    problem = Problem(
        sentence=fol.true(),
        cardinality_constraints=CardinalityConstraints(
            (
                LinearCardinalityConstraint(
                    (CardinalityTerm(relation),),
                    Comparator.EQ,
                    1,
                ),
            )
        ),
    )

    assert solve(
        problem,
        Domain(domain),
        algo=AlgoName.PROPOSITIONAL,
    ) == 4


def test_direct_propositional_counts_binary_evidence_without_reduction():
    from wfomc import Domain, Problem
    from wfomc.evidence import BinaryEvidence, Evidence, GroundBinaryLiteral
    from wfomc.fol import FOLContext

    fol = FOLContext()
    relation = fol.predicate("R", 2)
    left, right = (fol.constant(name) for name in ("a", "b"))
    problem = Problem(
        sentence=fol.true(),
        evidence=Evidence(
            binary=BinaryEvidence(
                (GroundBinaryLiteral(relation, left, right, True),)
            )
        ),
    )

    assert solve(
        problem,
        Domain(frozenset((left, right))),
        algo=AlgoName.PROPOSITIONAL,
    ) == 8


def test_direct_propositional_counts_unmentioned_ground_relation_entries():
    from wfomc import Domain, Problem
    from wfomc.fol import FOLContext

    fol = FOLContext()
    predicate = fol.predicate("P", 1)
    first, second = (fol.constant(name) for name in ("a", "b"))
    problem = Problem(
        sentence=predicate(first),
    )

    # P(a) is fixed true while the unmentioned P(b) remains free.
    assert solve(
        problem,
        Domain(frozenset((first, second))),
        algo=AlgoName.PROPOSITIONAL,
    ) == 2
