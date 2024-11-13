import pytest
import logging
import logzero

from pathlib import Path

from wfomc import wfomc, parse_input, Algo


current_path = Path(__file__).parent.absolute()
models_dir2args = {
    current_path.parent / 'models': (
        (Algo.STANDARD, ),
        (Algo.FAST, ),
        (Algo.FASTv2, ),
        (Algo.INCREMENTAL, ),
        (Algo.RECURSIVE, ),
    ),
    current_path.parent / 'models' / 'linear_order': (
        (Algo.INCREMENTAL, ),
        (Algo.RECURSIVE, ),
    ),
}
model_files = list((current_path.parent / 'models').glob('**/*.wfomcs')) + \
    list((current_path.parent / 'models').glob('**/*.mln'))
logzero.loglevel(logging.ERROR)


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
