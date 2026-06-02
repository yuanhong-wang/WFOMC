import pytest
import json

from pathlib import Path

from wfomc import wfomc, parse_input, Algo


current_path = Path(__file__).parent.absolute()
models_dir2args = {
    current_path.parent / 'models': (
        (Algo.STANDARD, ),
        (Algo.FAST, ),
        (Algo.FASTv2, ),
        (Algo.INCREMENTAL, ),
        (Algo.INCREMENTAL3, ),
        (Algo.RECURSIVE, ),
    ),
    current_path.parent / 'models' / 'linear_order': (
        (Algo.INCREMENTAL, ),
        (Algo.INCREMENTAL3, ),
        (Algo.RECURSIVE, ),
    ),
    current_path.parent / 'models' / 'predk': (
        (Algo.INCREMENTAL, ),
    ),
    current_path.parent / 'models' / 'regular_graphs': (
        (Algo.FASTv2, ),
        (Algo.RECURSIVE, ),
        (Algo.INCREMENTAL3, ),
    ),
    current_path.parent / 'models' / 'modk': (
        (Algo.INCREMENTAL3, ),
    )
}
model_files = list((current_path.parent / 'models/').glob('**/*.wfomcs')) + \
    list((current_path.parent / 'models').glob('*.mln'))
model_files = list(model_file for model_file in model_files if model_file.parent in models_dir2args)


@pytest.mark.parametrize(
    'model_file',
    [str(model_file) for model_file in model_files]
)
def test_model(model_file):
    results = list()
    for args in models_dir2args[Path(model_file).parent]:
        problem = parse_input(model_file)
        results.append(wfomc(problem, *args))
    assert all([r == results[0] for r in results])


answer_json = json.load(open(current_path.parent / 'models' / 'MATH' / 'all.json'))
MATH_files = list((current_path.parent / 'models' / 'MATH').glob('*.wfomcs'))
@pytest.mark.parametrize(
    'model_file, id',
    [(str(model_file), Path(model_file).stem) for model_file in MATH_files]
)
def test_MATH(model_file, id):
    problem = parse_input(str(model_file))
    problem_id = Path(model_file).stem
    answer = int(answer_json[problem_id]['answer'])
    res = wfomc(problem, algo=Algo.INCREMENTAL)
    assert res == answer, f"Failed MATH {problem_id}: {answer}(true) != {res}(computed)"
