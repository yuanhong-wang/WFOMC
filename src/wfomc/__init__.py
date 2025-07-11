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

from .algo import Algo
from .problems import WFOMCProblem, MLNProblem, MLN_to_WFOMC
from .parser import parse_input
from .parser.fol_parser import parse as fol_parse
from .fol.sc2 import SC2, to_sc2
from .fol.syntax import *
from .network.constraint import CardinalityConstraint
from .fol.utils import exactly_one, exactly_one_qf, exclusive_qf, exclusive
from .solver import wfomc
from .count_distribution import count_distribution
from .utils.polynomial import Rational, var, expand, \
    coeff_dict, coeff_monomial, round_rational
from .utils.third_typing import RingElement
from .utils.multinomial import MultinomialCoefficients, \
    multinomial, multinomial_less_than


__all__ = [
    'Algo',
    'WFOMCProblem',
    'MLNProblem',
    'MLN_to_WFOMC',
    'parse_input',
    'wfomc',
    'count_distribution',
    'SC2',
    'to_sc2',
    'fol_parse',
    'Rational',
    'var',
    'expand',
    'coeff_dict',
    'coeff_monomial',
    'round_rational',
    'RingElement',
    'MultinomialCoefficients',
    'multinomial',
    'multinomial_less_than',
    'exactly_one',
    'exactly_one_qf',
    'exclusive_qf',
    'exclusive',
    'Pred',
    'Term',
    'Var',
    'Const',
    'Formula',
    'QFFormula',
    'AtomicFormula',
    'Quantifier',
    'Universal',
    'Existential',
    'Counting',
    'QuantifiedFormula',
    'CompoundFormula',
    'Conjunction',
    'Disjunction',
    'Implication',
    'Equivalence',
    'Negation',
    'BinaryFormula',
    'SCOTT_PREDICATE_PREFIX',
    'AUXILIARY_PRED_NAME',
    'TSEITIN_PRED_NAME',
    'SKOLEM_PRED_NAME',
    'EVIDOM_PRED_NAME',
    'PREDS_FOR_EXISTENTIAL',
    'pretty_print',
    'X', 'Y', 'Z',
    'U', 'V', 'W',
    'top', 'bot',
    'CardinalityConstraint',
]


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


def main() -> None:
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
