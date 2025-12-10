import argparse
import logging
from contexttimer import Timer
from logzero import logger
import logzero

from wfomc.network import UnaryEvidenceEncoding
from wfomc.parser import parse_input
from wfomc.problems import WFOMCProblem
from wfomc.algo import Algo
from wfomc.solver import wfomc
from wfomc.utils import Rational
from wfomc.fol import Pred
from wfomc.utils import create_vars


def count_distribution(problem: WFOMCProblem, preds: list[Pred],
                       algo: Algo = Algo.STANDARD,
                       unary_evidence_encoding: UnaryEvidenceEncoding = UnaryEvidenceEncoding.CCS) \
        -> dict[tuple[int, ...], Rational]:
    pred2weight = {}
    pred2sym = {}
    weights = problem.weights
    syms = create_vars('c', len(preds))
    for sym, pred in zip(syms, preds):
        if pred in pred2weight:
            continue
        weight = weights.get(pred, (Rational(1, 1), Rational(1, 1)))
        pred2weight[pred] = (weight[0] * sym, weight[1])
        pred2sym[pred] = sym
    weights.update(pred2weight)
    problem = WFOMCProblem(
        problem.sentence, problem.domain, weights,
        problem.cardinality_constraint,
        problem.unary_evidence,
        problem.circle_len
    )

    res = wfomc(problem, algo=algo,
                unary_evidence_encoding=unary_evidence_encoding)

    symbols = [pred2sym[pred] for pred in preds]
    count_dist = {}
    for degrees, coef in coeff_dict(res, symbols):
        count_dist[degrees] = coef
    return count_dist


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
    # import sys
    # sys.setrecursionlimit(int(1e6))
    args = parse_args()
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')

    with Timer() as t:
        problem = parse_input(args.input)
    logger.info('Parse input: %ss', t)

    preds = problem.sentence.preds()

    count_dist = count_distribution(
        problem, preds, algo=args.algo
    )
    logger.info('Count distribution:')
    for counts, coef in count_dist.items():
        logger.info('Counts: %s, Coefficient: %s', counts, coef)


if __name__ == '__main__':
    main()
