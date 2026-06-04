"""
Cross-check the propositional (ganak-backed) counter against the lifted
algorithms. Skipped entirely when no ganak binary can be located, since the
counter is an external-tool integration.
"""
import os
from pathlib import Path

import pytest

_RUN_SLOW = os.environ.get('WFOMC_RUN_SLOW', '0') == '1'

from wfomc import Algo, parse_input, wfomc
from wfomc.algo import GanakError, LinearOrderEncoding, find_ganak
from wfomc.algo.PropositionalWFOMC import LINEAR_ORDER_ENCODING

# Some test cases are only slow when the FO³ axiomatization is the module
# default; under the default ``LinearOrderEncoding.PIN`` they finish in
# seconds and run unconditionally.
_AXIOMS_MODE = LINEAR_ORDER_ENCODING == LinearOrderEncoding.AXIOMS


_ROOT = Path(__file__).parent.parent
_IN_SCOPE_DIRS = [
    _ROOT / 'models',
    _ROOT / 'models' / 'unary_evidence',
    _ROOT / 'models' / 'linear_order',
    _ROOT / 'models' / 'linear_order_unary_evidence',
    _ROOT / 'models' / 'predk',
]
# A couple of CIRCULAR_PRED problems from MATH/ used to spot-check the
# circular-predecessor support. They are small enough to count quickly.
_CIRCULAR_MODELS: list[Path] = [
    _ROOT / 'models' / 'MATH' / '8.wfomcs',
    _ROOT / 'models' / 'MATH' / '33.wfomcs',
]


def _uses_multi_k_pred(problem) -> bool:
    """True iff the sentence uses PREDk for some k > 1."""
    for pred in problem.sentence.preds():
        if not pred.name.startswith('PRED') or pred.name == 'CIRCULAR_PRED':
            continue
        try:
            if int(pred.name[4:]) > 1:
                return True
        except ValueError:
            pass
    return False


def _collect_models() -> list[Path]:
    paths: list[Path] = []
    for d in _IN_SCOPE_DIRS:
        paths.extend(sorted(d.glob('*.wfomcs')))
        paths.extend(sorted(d.glob('*.mln')))
    return paths


def _ganak_available() -> bool:
    try:
        find_ganak()
        return True
    except GanakError:
        return False


@pytest.fixture(scope='session', autouse=True)
def _require_ganak():
    if not _ganak_available():
        pytest.skip('ganak binary not found; run `uv run wfomc-install-ganak`')


def _reference_algo(problem) -> Algo:
    """Pick the lifted reference algorithm that can handle the problem."""
    if problem.contain_linear_order_axiom():
        # fastv2 has no LEQ/PREk support; incremental handles both.
        return Algo.INCREMENTAL
    return Algo.FASTv2


_SLOW_UNDER_AXIOMS = {
    # n=15 with cardinality constraints + PRED1 FO³ definition -> ~hundreds
    # of ganak seconds in mode 3. Fine under the "pin" encoding.
    'models/predk/predecessor.wfomcs',
}


@pytest.mark.parametrize(
    'model_file', [str(p) for p in _collect_models()]
)
def test_propositional_matches_reference(model_file):
    """The propositional grounding must reproduce the lifted WFOMC."""
    relative = str(Path(model_file).relative_to(_ROOT))
    if _AXIOMS_MODE and relative in _SLOW_UNDER_AXIOMS and not _RUN_SLOW:
        pytest.skip(f'{relative} is slow under FO³ axiomatization; '
                    'set WFOMC_RUN_SLOW=1 or switch '
                    'LINEAR_ORDER_ENCODING to "pin"')
    problem = parse_input(model_file)
    if _uses_multi_k_pred(problem):
        pytest.skip('PREDk for k > 1 is out of scope for the propositional '
                    'counter (only PRED and CIRCULAR_PRED are supported)')
    ref_algo = _reference_algo(problem)
    reference = wfomc(problem, algo=ref_algo)
    propositional = wfomc(problem, algo=Algo.PROPOSITIONAL)
    assert propositional == reference, (
        f'propositional={propositional} != {ref_algo}={reference} '
        f'for {model_file}'
    )


@pytest.mark.skipif(
    _AXIOMS_MODE and not _RUN_SLOW,
    reason='Circular axiomatization is slow (~5-10 min on n=10); '
           'set WFOMC_RUN_SLOW=1 to enable, or switch '
           'LINEAR_ORDER_ENCODING to "pin".',
)
@pytest.mark.parametrize(
    'model_file', [str(p) for p in _CIRCULAR_MODELS if p.exists()]
)
def test_propositional_matches_reference_circular(model_file):
    """The propositional grounding must reproduce the lifted WFOMC on
    CIRCULAR_PRED problems too. Under the FO³ ``axioms`` encoding these
    take minutes; under the default ``pin`` encoding they run in seconds.
    """
    problem = parse_input(model_file)
    reference = wfomc(problem, algo=Algo.INCREMENTAL)
    propositional = wfomc(problem, algo=Algo.PROPOSITIONAL)
    assert propositional == reference, (
        f'propositional={propositional} != incremental={reference} '
        f'for {model_file}'
    )
