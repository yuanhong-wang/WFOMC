"""Cross-check the GANAK-backed counter against lifted algorithms."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wfomc import AlgoName, parse_input, solve
from wfomc.algo import LinearOrderEncoding, resolve_linear_order_encoding
from wfomc.ganak import GanakError, find_ganak
from wfomc.engine.features import analyze_features


RUN_SLOW = os.environ.get("WFOMC_RUN_SLOW", "0") == "1"
AXIOMS_MODE = resolve_linear_order_encoding(None) is LinearOrderEncoding.AXIOMS
ROOT = Path(__file__).parents[2]
IN_SCOPE_DIRS = (
    ROOT / "models",
    ROOT / "models" / "unary_evidence",
    ROOT / "models" / "linear_order",
    ROOT / "models" / "linear_order" / "unary_evidence",
    ROOT / "models" / "linear_order" / "predk",
)
CIRCULAR_MODELS = (
    ROOT / "models" / "MATH" / "8.wfomcs",
    ROOT / "models" / "MATH" / "33.wfomcs",
)
SLOW_UNDER_AXIOMS = frozenset(
    ("models/linear_order/predk/predecessor.wfomcs",)
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
        for order, _predicate in analyze_features(problem).predecessor_predicates
    )


def _reference_algo(problem) -> AlgoName:
    features = analyze_features(problem)
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
    problem = parse_input(str(model_file))
    if _uses_multi_k_pred(problem):
        pytest.skip("PREDk for k > 1 is outside propositional scope")

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
    problem = parse_input(str(model_file))

    reference = solve(problem, algo=AlgoName.INCREMENTAL)
    propositional = solve(problem, algo=AlgoName.PROPOSITIONAL)

    assert propositional == reference
