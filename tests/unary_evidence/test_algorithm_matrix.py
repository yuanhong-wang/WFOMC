from pathlib import Path

import pytest

from wfomc import Algo, UnaryEvidenceStrategy, parse_input, wfomc
from wfomc.parser.wfomcs_parser import parse as parse_wfomcs
from wfomc.solver import resolve_unary_evidence_strategy


ROOT = Path(__file__).parents[2]
UNARY_EVIDENCE_MODELS = sorted(
    (ROOT / "models" / "unary_evidence").glob("*.wfomcs")
) + sorted((ROOT / "models" / "unary_evidence").glob("*.mln"))

ALGORITHMS = (
    Algo.STANDARD,
    Algo.FAST,
    Algo.FASTv2,
    Algo.INCREMENTAL,
    Algo.INCREMENTAL3,
    Algo.RECURSIVE,
)


@pytest.mark.parametrize(
    "strategy",
    (UnaryEvidenceStrategy.AUTO, UnaryEvidenceStrategy.CCS),
)
@pytest.mark.parametrize("model_file", UNARY_EVIDENCE_MODELS, ids=lambda path: path.stem)
def test_unary_evidence_algorithm_matrix(model_file, strategy):
    results = [
        wfomc(parse_input(str(model_file)), algo, strategy)
        for algo in ALGORITHMS
    ]

    assert all(result == results[0] for result in results)


@pytest.mark.parametrize(
    "name, expected",
    (
        ("evidence-only.wfomcs", 4),
        ("impossible-evidence.wfomcs", 0),
        ("overlapping-profiles.wfomcs", 36),
    ),
)
def test_named_unary_evidence_regressions(name, expected):
    problem = parse_input(str(ROOT / "models" / "unary_evidence" / name))

    assert wfomc(problem, Algo.STANDARD) == expected
    assert wfomc(problem, Algo.STANDARD, UnaryEvidenceStrategy.CCS) == expected


@pytest.mark.parametrize(
    "algo, expected",
    (
        (Algo.STANDARD, UnaryEvidenceStrategy.AUTO),
        (Algo.FAST, UnaryEvidenceStrategy.CCS),
        (Algo.FASTv2, UnaryEvidenceStrategy.AUTO),
        (Algo.INCREMENTAL, UnaryEvidenceStrategy.AUTO),
        (Algo.INCREMENTAL3, UnaryEvidenceStrategy.AUTO),
        (Algo.RECURSIVE, UnaryEvidenceStrategy.CCS),
    ),
)
def test_auto_resolves_to_best_supported_strategy(algo, expected):
    assert resolve_unary_evidence_strategy(
        algo, UnaryEvidenceStrategy.AUTO
    ) == expected


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_explicit_ccs_is_preserved_for_every_algorithm(algo):
    assert resolve_unary_evidence_strategy(
        algo, UnaryEvidenceStrategy.CCS
    ) == UnaryEvidenceStrategy.CCS


def test_named_constants_in_sentence_fail_fast_with_unary_evidence():
    problem = parse_wfomcs(r"""
\forall X: (R(X, domain0))

domain = {domain0, domain1}

P(domain0)
""")

    with pytest.raises(ValueError, match="exchangeable domain"):
        wfomc(problem, Algo.INCREMENTAL)
