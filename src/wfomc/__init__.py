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
