from itertools import product
import pytest
import json
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
    ),
    current_path.parent / 'models' / 'predk': (
        (Algo.INCREMENTAL, ),
    ),
}
model_files = list((current_path.parent / 'models').glob('*.wfomcs')) + \
    list((current_path.parent / 'models').glob('*.mln'))
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


# answer_json = json.load(open(current_path.parent / 'models' / 'MATH' / 'all.json'))
# MATH_files = list((current_path.parent / 'models' / 'MATH').glob('*.wfomcs'))
# @pytest.mark.parametrize(
#     'model_file, id',
#     [(str(model_file), Path(model_file).stem) for model_file in MATH_files]
# )
# def test_MATH(model_file, id):
#     problem = parse_input(str(model_file))
#     problem_id = Path(model_file).stem
#     answer = int(answer_json[problem_id]['answer'])
#     res = wfomc(problem, algo=Algo.INCREMENTAL, unary_evidence_encoding=UnaryEvidenceEncoding.CCS)
#     assert res == answer, f"Failed CCS for MATH {problem_id}: {answer}(true) != {res}(computed)"
#     res = wfomc(problem, algo=Algo.INCREMENTAL, unary_evidence_encoding=UnaryEvidenceEncoding.PC)
#     assert res == answer, f"Failed PC for MATH {problem_id}: {answer}(true) != {res}(computed)"
