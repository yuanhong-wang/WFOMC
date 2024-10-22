from __future__ import annotations
from collections import defaultdict

from enum import Enum
from functools import reduce
import math
import os
import argparse
import logging
import logzero

from logzero import logger
from typing import Callable
from contexttimer import Timer

from counting_fo2.cell_graph.cell_graph import build_cell_graphs
from counting_fo2.problems import WFOMCSProblem

from counting_fo2.utils import MultinomialCoefficients, multinomial, \
    multinomial_less_than, RingElement, Rational, round_rational
from counting_fo2.cell_graph import CellGraph, Cell
from counting_fo2.context import WFOMCContext
from counting_fo2.parser import parse_input
from counting_fo2.fol.syntax import Const, Pred, QFFormula, PREDS_FOR_EXISTENTIAL
from counting_fo2.utils.polynomial import coeff_dict, create_vars, expand


class Algo(Enum):
    STANDARD = 'standard'
    FASTER = 'faster'
    FASTERv2 = 'fasterv2'
    INCREMENTAL = 'incremental'

    def __str__(self):
        return self.value


def get_config_weight_standard_faster(config: list[int],
                                      cell_weights: list[RingElement],
                                      edge_weights: list[list[RingElement]]) \
        -> RingElement:
    res = Rational(1, 1)
    for i, n_i in enumerate(config):
        if n_i == 0:
            continue
        n_i = Rational(n_i, 1)
        res *= cell_weights[i] ** n_i
        res *= edge_weights[i][i] ** (n_i * (n_i - 1) // Rational(2, 1))
        for j, n_j in enumerate(config):
            if j <= i:
                continue
            if n_j == 0:
                continue
            n_j = Rational(n_j, 1)
            res *= edge_weights[i][j] ** (n_i * n_j)
    return res


def get_config_weight_standard(cell_graph: CellGraph,
                               cell_config: dict[Cell, int]) -> RingElement:
    res = Rational(1, 1)
    for i, (cell_i, n_i) in enumerate(cell_config.items()):
        if n_i == 0:
            continue
        res = res * cell_graph.get_cell_weight(cell_i) ** n_i
        res = res * cell_graph.get_two_table_weight(
            (cell_i, cell_i)
        ) ** (n_i * (n_i - 1) // 2)
        for j, (cell_j, n_j) in enumerate(cell_config.items()):
            if j <= i:
                continue
            if n_j == 0:
                continue
            res = res * cell_graph.get_two_table_weight(
                (cell_i, cell_j)
            ) ** (n_i * n_j)
    # logger.debug('Config weight: %s', res)
    return res


def faster_wfomc(formula: QFFormula,
                 domain: set[Const],
                 get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                 modified_cell_symmetry: bool = False) -> RingElement:
    domain_size = len(domain)
    res = Rational(0, 1)
    for opt_cell_graph, weight in build_cell_graphs(
        formula, get_weight, optimized=True,
        domain_size=domain_size,
        modified_cell_symmetry=modified_cell_symmetry
    ):
        cliques = opt_cell_graph.cliques
        nonind = opt_cell_graph.nonind
        i2_ind = opt_cell_graph.i2_ind
        nonind_map = opt_cell_graph.nonind_map

        res_ = Rational(0, 1)
        with Timer() as t:
            for partition in multinomial_less_than(len(nonind), domain_size):
                mu = tuple(partition)
                if sum(partition) < domain_size:
                    mu = mu + (domain_size - sum(partition),)
                coef = MultinomialCoefficients.coef(mu)
                body = Rational(1, 1)

                for i, clique1 in enumerate(cliques):
                    for j, clique2 in enumerate(cliques):
                        if i in nonind and j in nonind:
                            if i < j:
                                body = body * opt_cell_graph.get_two_table_weight(
                                    (clique1[0], clique2[0])
                                ) ** (partition[nonind_map[i]] *
                                      partition[nonind_map[j]])

                for l in nonind:
                    body = body * opt_cell_graph.get_J_term(
                        l, partition[nonind_map[l]]
                    )
                    if not modified_cell_symmetry:
                        body = body * opt_cell_graph.get_cell_weight(
                            cliques[l][0]
                        ) ** partition[nonind_map[l]]

                opt_cell_graph.setup_term_cache()
                mul = opt_cell_graph.get_term(len(i2_ind), 0, partition)
                res_ = res_ + coef * mul * body
        res = res + weight * res_
    logger.info('WFOMC time: %s', t.elapsed)
    return res


def standard_wfomc(formula: QFFormula,
                   domain: set[Const],
                   get_weight: Callable[[Pred], tuple[RingElement, RingElement]]) -> RingElement:
    # cell_graph.show()
    res = Rational(0, 1)
    domain_size = len(domain)
    for cell_graph, weight in build_cell_graphs(formula, get_weight):
        res_ = Rational(0, 1)
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        for partition in multinomial(n_cells, domain_size):
            coef = MultinomialCoefficients.coef(partition)
            cell_config = dict(zip(cells, partition))
            # logger.debug(
            #     '=' * 15 + ' Compute WFOMC for the partition %s ' + '=' * 15,
            #     dict(filter(lambda x: x[1] != 0, cell_config.items())
            # ))
            res_ = res_ + coef * get_config_weight_standard(
                cell_graph, cell_config
            )
        res = res + weight * res_
    return res


def incremental_wfomc(formula: QFFormula,
                      domain: set[Const],
                      get_weight: Callable[[Pred],
                                           tuple[RingElement, RingElement]],
                      leq_pred: Pred = None) -> RingElement:
    res = Rational(0, 1)
    domain_size = len(domain)
    for cell_graph, weight in build_cell_graphs(
            formula, get_weight, leq_pred=leq_pred
    ):
        # cell_graph.show()
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        domain_size = len(domain)

        table = dict(
            (tuple(int(k == i) for k in range(n_cells)),
             cell_graph.get_cell_weight(cell))
            for i, cell in enumerate(cells)
        )
        for _ in range(domain_size - 1):
            old_table = table
            table = dict()
            for j, cell in enumerate(cells):
                w = cell_graph.get_cell_weight(cell)
                for ivec, w_old in old_table.items():
                    w_new = w_old * w * reduce(
                        lambda x, y: x * y,
                        (
                            cell_graph.get_two_table_weight((cell, cells[k]))
                            ** int(ivec[k]) for k in range(n_cells)
                        ),
                        Rational(1, 1)
                    )
                    ivec = list(ivec)
                    ivec[j] += 1
                    ivec = tuple(ivec)

                    w_new = w_new + table.get(ivec, Rational(0, 1))
                    table[tuple(ivec)] = w_new
        res = res + weight * sum(table.values())

    # if leq_pred is not None:
    #     res *= Rational(math.factorial(domain_size), 1)
    return res


def precompute_ext_weight(cell_graph: CellGraph, domain_size: int,
                          context: WFOMCContext) \
        -> dict[frozenset[tuple[Cell, frozenset[Pred], int]], RingElement]:
    existential_weights = defaultdict(lambda: Rational(0, 1))
    cells = cell_graph.get_cells()
    eu_configs = []
    for cell in cells:
        config = []
        for domain_pred, tseitin_preds in context.domain_to_evidence_preds.items():
            if cell.is_positive(domain_pred):
                config.append((
                    cell.drop_preds(
                        prefixes=PREDS_FOR_EXISTENTIAL), tseitin_preds
                ))
        eu_configs.append(config)

    cell_weights, edge_weights = cell_graph.get_all_weights()

    for partition in multinomial(len(cells), domain_size):
        # res = get_config_weight_standard(
        #     cell_graph, dict(zip(cells, partition))
        # )
        res = get_config_weight_standard_faster(
            partition, cell_weights, edge_weights
        )
        eu_config = defaultdict(lambda: 0)
        for idx, n in enumerate(partition):
            for config in eu_configs[idx]:
                eu_config[config] += n
        eu_config = dict(
            (k, v) for k, v in eu_config.items() if v > 0
        )
        existential_weights[
            frozenset((*k, v) for k, v in eu_config.items())
        ] += (Rational(MultinomialCoefficients.coef(partition), 1) * res)
    # remove duplications
    for eu_config in existential_weights.keys():
        dup_factor = Rational(MultinomialCoefficients.coef(
            tuple(c[2] for c in eu_config)
        ), 1)
        existential_weights[eu_config] /= dup_factor
    return existential_weights


def wfomc(problem: WFOMCSProblem, algo: Algo = Algo.STANDARD) -> Rational:
    # both standard and faster WFOMCs need precomputation
    if algo == Algo.STANDARD or algo == Algo.FASTER or \
            algo == algo.FASTERv2:
        MultinomialCoefficients.setup(len(problem.domain))

    context = WFOMCContext(problem)
    leq_pred = Pred('LEQ', 2)
    if leq_pred in context.formula.preds():
        logger.info('Linear order axiom with the predicate LEQ is found')
        logger.info('Invoke incremental WFOMC')
        algo = Algo.INCREMENTAL
    else:
        leq_pred = None

    with Timer() as t:
        if algo == Algo.STANDARD:
            res = standard_wfomc(
                context.formula, context.domain, context.get_weight
            )
        elif algo == Algo.FASTER:
            res = faster_wfomc(
                context.formula, context.domain, context.get_weight
            )
        elif algo == Algo.FASTERv2:
            res = faster_wfomc(
                context.formula, context.domain, context.get_weight, True
            )
        elif algo == Algo.INCREMENTAL:
            res = incremental_wfomc(
                context.formula, context.domain,
                context.get_weight, leq_pred
            )
    res = context.decode_result(res)
    logger.info('WFOMC time: %s', t.elapsed)
    return res


def count_distribution(problem: WFOMCSProblem, preds: list[Pred],
                       algo: Algo = Algo.STANDARD) \
        -> dict[tuple[int, ...], Rational]:
    context = WFOMCContext(problem)
    # both standard and faster WFOMCs need precomputation
    if algo == Algo.STANDARD or algo == Algo.FASTER or \
            algo == algo.FASTERv2:
        MultinomialCoefficients.setup(len(problem.domain))
    leq_pred = Pred('LEQ', 2)
    if leq_pred in context.formula.preds():
        logger.info('Linear order axiom with the predicate LEQ is found')
        logger.info('Invoke incremental WFOMC')
        algo = Algo.INCREMENTAL
    else:
        leq_pred = None

    pred2weight = {}
    pred2sym = {}
    syms = create_vars('x0:{}'.format(len(preds)))
    for sym, pred in zip(syms, preds):
        if pred in pred2weight:
            continue
        weight = context.get_weight(pred)
        pred2weight[pred] = (weight[0] * sym, weight[1])
        pred2sym[pred] = sym
    context.weights.update(pred2weight)

    if algo == Algo.STANDARD:
        res = standard_wfomc(
            context.formula, context.domain, context.get_weight
        )
    elif algo == Algo.FASTER:
        res = faster_wfomc(
            context.formula, context.domain, context.get_weight
        )
    elif algo == Algo.FASTERv2:
        res = faster_wfomc(
            context.formula, context.domain, context.get_weight, True
        )
    elif algo == Algo.INCREMENTAL:
        res = incremental_wfomc(
            context.formula, context.domain,
            context.get_weight, leq_pred
        )

    symbols = [pred2sym[pred] for pred in preds]
    count_dist = {}
    res = expand(res)
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
                        choices=list(Algo), default=Algo.FASTER)
    parser.add_argument('--domain_recursive',
                        action='store_true', default=False,
                        help='use domain recursive algorithm '
                             '(only for existential quantified MLN)')
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
