"""Published MATH answer regressions."""

import json
import os
from pathlib import Path

import pytest

from wfomc import AlgoName, parse_input, solve
from wfomc.engine.features import analyze_features


ROOT = Path(__file__).parents[2]
RUN_SLOW = os.environ.get("WFOMC_RUN_SLOW", "0") == "1"
MATH_ANSWERS = json.loads((ROOT / "models" / "MATH" / "all.json").read_text())
MATH_FILES = tuple(sorted((ROOT / "models" / "MATH").glob("*.wfomcs")))


@pytest.mark.parametrize("model_file", MATH_FILES, ids=lambda path: path.stem)
def test_math_published_answers(model_file: Path):
    problem = parse_input(str(model_file))
    expected = int(MATH_ANSWERS[model_file.stem]["answer"])
    if RUN_SLOW:
        algorithm = AlgoName.INCREMENTAL
    else:
        features = analyze_features(problem)
        algorithm = (
            AlgoName.INCREMENTAL
            if features.has_predk or features.has_circular_pred
            else AlgoName.INCREMENTAL3
        )

    assert solve(problem, algo=algorithm) == expected
