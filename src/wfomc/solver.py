import os
import sys
import argparse
from loguru import logger
from contexttimer import Timer

from wfomc.network import UnaryEvidenceEncoding
from wfomc.problems import WFOMCProblem
from wfomc.algo import Algo, standard_wfomc, fast_wfomc, incremental_wfomc, recursive_wfomc
from wfomc.utils import MultinomialCoefficients, Rational, round_rational, Poly
from wfomc.context import WFOMCContext
from wfomc.parser import parse_input

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def wfomc(problem: WFOMCProblem, algo: Algo = Algo.STANDARD,
          unary_evidence_encoding: UnaryEvidenceEncoding = UnaryEvidenceEncoding.CCS,
          debug: bool = False) -> Rational:
    level = "DEBUG" if debug else "INFO"
    _handler_id = logger.add(
        sys.stderr, level=level, filter="wfomc", colorize=True, format=_LOG_FORMAT,
    )
    logger.enable("wfomc")
    try:
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
        logger.info('WFOMC time: {}', t.elapsed)
        return res
    finally:
        logger.remove(_handler_id)
        logger.disable("wfomc")


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
    parser.add_argument('--unary_evidence_encoding', '-e', type=UnaryEvidenceEncoding,
                        choices=list(UnaryEvidenceEncoding),
                        default=UnaryEvidenceEncoding.CCS)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    level = "DEBUG" if args.debug else "INFO"

    # Remove the default loguru stderr handler; parse_input() and wfomc()
    # each add their own scoped handler.
    try:
        logger.remove(0)
    except ValueError:
        pass

    # File sink covers the entire session. It receives records whenever
    # parse_input() or wfomc() call logger.enable("wfomc").
    logger.add(
        f'{args.output_dir}/log.txt',
        mode='w',
        level=level,
        filter="wfomc",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    )

    with Timer() as t:
        problem = parse_input(args.input, debug=args.debug)
    print(f'Parse input: {t.elapsed:.4f}s')

    res = wfomc(
        problem, algo=args.algo,
        unary_evidence_encoding=args.unary_evidence_encoding,
        debug=args.debug,
    )

    print(f'WFOMC (arbitrary precision): {res}')
    if isinstance(res, Rational):
        round_val = round_rational(res)
        print(f'WFOMC (round): {round_val} (exp({round_val.ln()}))')
