from __future__ import annotations

import os
import argparse
import logging
import logzero

from logzero import logger
from contexttimer import Timer

from wfomc.problems import WFOMCProblem
from wfomc.algo import Algo, standard_wfomc, fast_wfomc, incremental_wfomc, recursive_wfomc

from wfomc.utils import MultinomialCoefficients, Rational, round_rational
from wfomc.context import WFOMCContext
from wfomc.parser import parse_input
from wfomc.fol.syntax import Pred


def wfomc(problem: WFOMCProblem, algo: Algo = Algo.STANDARD) -> Rational:
    # both standard and fast WFOMCs need precomputation
    if algo == Algo.STANDARD or algo == Algo.FAST or \
            algo == algo.FASTv2:
        MultinomialCoefficients.setup(len(problem.domain))

    if problem.contain_linear_order_axiom():
        logger.info('Linear order axiom with the predicate LEQ is found')
        if algo != Algo.INCREMENTAL and algo != Algo.RECURSIVE:
            raise RuntimeError("Linear order axiom is only supported by the "
                               "incremental and recursive WFOMC algorithms")

    logger.info(f'Invoke WFOMC with {algo} algorithm')

    context = WFOMCContext(problem)
    with Timer() as t:
        if algo == Algo.STANDARD:
            res = standard_wfomc(context)
        elif algo == Algo.FAST:
            res = fast_wfomc(context)
        elif algo == Algo.FASTv2:
            res = fast_wfomc(context, True)
        elif algo == Algo.INCREMENTAL:
            res = incremental_wfomc(context)
        elif algo == Algo.RECURSIVE:
            res = recursive_wfomc(context)
    res = context.decode_result(res)
    logger.info('WFOMC time: %s', t.elapsed)
    return res
