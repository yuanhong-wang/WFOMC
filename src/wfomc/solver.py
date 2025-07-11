from __future__ import annotations

import os
import argparse
import logging
import logzero

from logzero import logger
from contexttimer import Timer
from wfomc.network.constraint import UnaryEvidenceEncoding

from wfomc.problems import WFOMCProblem
from wfomc.algo import Algo, standard_wfomc, fast_wfomc, incremental_wfomc, recursive_wfomc

from wfomc.utils import MultinomialCoefficients, Rational, round_rational
from wfomc.context import WFOMCContext
from wfomc.parser import parse_input
from wfomc.utils.polynomial import expand


def wfomc(problem: WFOMCProblem, algo: Algo = Algo.STANDARD,
          unary_evidence_encoding: UnaryEvidenceEncoding = UnaryEvidenceEncoding.CCS) -> Rational:
    MultinomialCoefficients.setup(len(problem.domain))

    if problem.contain_linear_order_axiom():
        logger.info('Linear order axiom with the predicate LEQ is found')
        if algo != Algo.INCREMENTAL and algo != Algo.RECURSIVE:
            raise RuntimeError("Linear order axiom is only supported by the "
                               "incremental and recursive WFOMC algorithms")
    if problem.contain_predecessor_axiom():
        logger.info('Predecessor predicate PRED is found')
        if algo != Algo.INCREMENTAL:
            raise RuntimeError("Predecessor axiom is only supported by the "
                               "incremental WFOMC algorithm")

    if problem.contain_unary_evidence():
        logger.info(f'Unary evidence is found, using {unary_evidence_encoding} encoding')
        if unary_evidence_encoding == UnaryEvidenceEncoding.PC and \
                algo != Algo.FASTv2 and algo != Algo.INCREMENTAL:
            raise RuntimeError("Partition constraint is only supported for the "
                               "fastv2 WFOMC and incremental WFOMC algorithms")

    logger.info(f'Invoke WFOMC with {algo} algorithm and {unary_evidence_encoding} encoding')

    context = WFOMCContext(problem, unary_evidence_encoding)
    with Timer() as t:
        if algo == Algo.STANDARD:
            res = standard_wfomc(context)
        elif algo == Algo.FAST:
            res = fast_wfomc(context)
        elif algo == Algo.FASTv2:
            res = fast_wfomc(context, True)
        elif algo == Algo.INCREMENTAL:
            res = incremental_wfomc(context, problem.circle_len)
        elif algo == Algo.RECURSIVE:
            res = recursive_wfomc(context)
    res = context.decode_result(res)
    logger.info('WFOMC time: %s', t.elapsed)
    return res
