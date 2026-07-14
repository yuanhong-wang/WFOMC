"""Cross-algorithm checks over the repository's representative model families."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wfomc import AlgoName, parse_input, solve
from wfomc.engine.features import analyze_features


ROOT = Path(__file__).parents[2]
RUN_SLOW = os.environ.get("WFOMC_RUN_SLOW", "0") == "1"
MODEL_DIRS_TO_ALGOS = {
    ROOT / "models": (
        AlgoName.STANDARD,
        AlgoName.FAST,
        AlgoName.FASTV2,
        AlgoName.INCREMENTAL,
        AlgoName.INCREMENTAL3,
        AlgoName.RECURSIVE,
    ),
    ROOT / "models" / "linear_order": (
        AlgoName.INCREMENTAL,
        AlgoName.INCREMENTAL3,
        AlgoName.RECURSIVE,
    ),
    ROOT / "models" / "linear_order" / "predk": (AlgoName.INCREMENTAL,),
    ROOT / "models" / "linear_order" / "unary_evidence": (
        AlgoName.INCREMENTAL,
        AlgoName.INCREMENTAL3,
        AlgoName.RECURSIVE,
    ),
    ROOT / "models" / "regular_graphs": (
        AlgoName.FASTV2,
        AlgoName.RECURSIVE,
        AlgoName.INCREMENTAL3,
    ),
    ROOT / "models" / "modk": (AlgoName.INCREMENTAL3,),
}
MODEL_FILES = tuple(
    model
    for model in (
        *sorted((ROOT / "models").glob("**/*.wfomcs")),
        *sorted((ROOT / "models").glob("*.mln")),
    )
    if model.parent in MODEL_DIRS_TO_ALGOS
)


@pytest.mark.parametrize("model_file", MODEL_FILES, ids=lambda path: path.stem)
def test_model_family_algorithms_agree(model_file: Path):
    problem = parse_input(str(model_file))
    algorithms = MODEL_DIRS_TO_ALGOS[model_file.parent]
    features = analyze_features(problem)
    if (
        not RUN_SLOW
        and AlgoName.INCREMENTAL3 in algorithms
        and (features.has_c2_counting or features.has_linear_order)
    ):
        algorithms = (AlgoName.INCREMENTAL3,)

    results = {algo: solve(problem, algo=algo) for algo in algorithms}

    first = next(iter(results.values()))
    assert all(result == first for result in results.values()), results
