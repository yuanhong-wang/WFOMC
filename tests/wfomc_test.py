import pytest
import logging
import logzero

from pathlib import Path

from counting_fo2.parser import parse_input
from counting_fo2.wfomc import Algo, wfomc

current_path = Path(__file__).parent.absolute()
model_files = (current_path.parent / 'models').glob('*')
algos = [Algo.STANDARD, Algo.FASTER, Algo.FASTERv2]
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
