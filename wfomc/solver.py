from __future__ import annotations

import os
import argparse
import logging
import logzero

from logzero import logger
from contexttimer import Timer
from wfomc.algo.FastWFOMC import fast_wfomc_with_pc

from wfomc.problems import WFOMCProblem
from wfomc.algo import Algo, standard_wfomc, fast_wfomc, incremental_wfomc

from wfomc.utils import MultinomialCoefficients, Rational, round_rational
from wfomc.context import WFOMCContext
from wfomc.parser import parse_input
from wfomc.fol.syntax import Pred


def wfomc(problem: WFOMCProblem, algo: Algo = Algo.STANDARD,
          use_partition_constraint: bool = False) -> Rational:
    if not problem.unary_evidence:
        logger.info('No unary evidence is found, use_partition_constraint is ignored')
        use_partition_constraint = False
    if use_partition_constraint:
        logger.info("Use partition constraint to handle unary evidence, "
                    "only support the algorithm of fast WFOMC")
        algo = Algo.FASTv2
    # both standard and fast WFOMCs need precomputation
    if algo == Algo.STANDARD or algo == Algo.FAST or \
            algo == algo.FASTv2:
        MultinomialCoefficients.setup(len(problem.domain))

    context = WFOMCContext(problem, use_partition_constraint)
    leq_pred = Pred('LEQ', 2)
    if leq_pred in context.formula.preds():
        logger.info('Linear order axiom with the predicate LEQ is found')
        logger.info('Invoke incremental WFOMC')
        if use_partition_constraint:
            logger.info("Partition constraint is not supported for the "
                        "incremental WFOMC, ignore the flag")
        algo = Algo.INCREMENTAL
    else:
        leq_pred = None

    with Timer() as t:
        if algo == Algo.STANDARD:
            res = standard_wfomc(
                context.formula, context.domain, context.get_weight
            )
        elif algo == Algo.FAST:
            res = fast_wfomc(
                context.formula, context.domain, context.get_weight
            )
        elif algo == Algo.FASTv2:
            if use_partition_constraint:
                res = fast_wfomc_with_pc(
                    context.formula, context.domain,
                    context.get_weight,
                    context.partition_constraint,
                )
            else:
                res = fast_wfomc(
                    context.formula,
                    context.domain,
                    context.get_weight,
                    True
                )
        elif algo == Algo.INCREMENTAL:
            res = incremental_wfomc(
                context.formula, context.domain,
                context.get_weight, leq_pred
            )
    res = context.decode_result(res)
    logger.info('WFOMC time: %s', t.elapsed)
    return res


def parse_args():
    parser = argparse.ArgumentParser(
        description='WFOMC for MLN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='./check-points')
    parser.add_argument('--algo', '-a', type=Algo,
                        choices=list(Algo), default=Algo.FASTv2)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # import sys
    # sys.setrecursionlimit(int(1e6))
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')

    with Timer() as t:
        problem = parse_input(args.input)
    logger.info('Parse input: %ss', t)

    res = wfomc(
        problem, algo=args.algo
    )
    logger.info('WFOMC (arbitrary precision): %s', res)
    round_val = round_rational(res)
    logger.info('WFOMC (round): %s (exp(%s))', round_val, round_val.ln())
