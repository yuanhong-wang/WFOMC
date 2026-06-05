import pytest
import json

from pathlib import Path

from sympy import symbols

from wfomc import (
    Algo,
    Const,
    Pred,
    Rational,
    WFOMCProblem,
    WFOMCResult,
    fol_parse,
    parse_input,
    to_sc2,
    wfomc,
)
from wfomc.parser.wfomcs_parser import parse as parse_wfomcs_text


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


def test_wfomc_returns_public_result_wrapper():
    problem = parse_input(str(current_path.parent / "models" / "2-colored-graph.wfomcs"))
    result = wfomc(problem, Algo.FASTv2)
    assert isinstance(result, WFOMCResult)
    assert result.is_constant()
    assert result.constant_value() is not None


def test_wfomc_result_exposes_projected_polynomial_terms():
    x = symbols("x")
    sentence = to_sc2(fol_parse(r"\forall X: (P(X))"))
    problem = WFOMCProblem(
        sentence,
        {Const("a"), Const("b")},
        {Pred("P", 1): (x, Rational(1, 1))},
    )

    result = wfomc(problem, Algo.FASTv2)

    assert result.is_polynomial()
    assert dict(result.terms([x])) == {(2,): Rational(1, 1)}


def test_unsatisfiable_wfomc_returns_zero():
    problem = parse_wfomcs_text(r"""
\forall X: ((S(X) -> C(X)) & ~(S(X) & C(X)))
domain = {e_1, e_10, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9}

|C| = 5

S(e_1), S(e_10), S(e_2), S(e_3), S(e_4), S(e_5), S(e_6), S(e_7), S(e_8), S(e_9)
""")

    result = wfomc(problem, Algo.FASTv2)

    assert isinstance(result, WFOMCResult)
    assert result == 0


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
