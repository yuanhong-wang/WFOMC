import pytest
import logging
import logzero

from pathlib import Path

from wfomc.parser import parse_input
from wfomc import wfomc
from wfomc.algo import Algo

current_path = Path(__file__).parent.absolute()
model_files = (current_path.parent / 'models').glob('*')
algos = [Algo.STANDARD, Algo.FAST, Algo.FASTv2]
logzero.loglevel(logging.ERROR)


def wfomc_proxy(model_file, algo):
    problem = parse_input(model_file)
    return wfomc(problem, algo)


@pytest.mark.parametrize(
    'model_file',
    [str(model_file) for model_file in model_files]
)
def test_model(model_file):
    results = list()
    for algo in algos:
        results.append(wfomc_proxy(model_file, algo))
    print(results)
    assert all([r == results[0] for r in results])
