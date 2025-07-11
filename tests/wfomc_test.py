from itertools import product
import pytest
import logging
import logzero

from pathlib import Path

from wfomc import wfomc, parse_input, Algo, UnaryEvidenceEncoding


current_path = Path(__file__).parent.absolute()
models_dir2args = {
    current_path.parent / 'models': (
        (Algo.STANDARD, ),
        (Algo.FAST, ),
        (Algo.FASTv2, ),
        (Algo.INCREMENTAL, ),
        (Algo.RECURSIVE, ),
    ),
    current_path.parent / 'models' / 'unary_evidence': (
        (Algo.STANDARD, UnaryEvidenceEncoding.CCS),
        (Algo.FAST, UnaryEvidenceEncoding.CCS),
        (Algo.FASTv2, UnaryEvidenceEncoding.CCS),
        (Algo.INCREMENTAL, UnaryEvidenceEncoding.CCS),
        (Algo.RECURSIVE, UnaryEvidenceEncoding.CCS),
        (Algo.INCREMENTAL, UnaryEvidenceEncoding.PC),
        (Algo.FASTv2, UnaryEvidenceEncoding.PC),
    ),
    current_path.parent / 'models' / 'linear_order': (
        (Algo.INCREMENTAL, ),
        (Algo.RECURSIVE, ),
    ),
    current_path.parent / 'models' / 'linear_order_unary_evidence': (
        (Algo.INCREMENTAL, UnaryEvidenceEncoding.CCS),
        (Algo.RECURSIVE, UnaryEvidenceEncoding.CCS),
        (Algo.INCREMENTAL, UnaryEvidenceEncoding.PC),
    )
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
