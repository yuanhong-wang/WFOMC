from __future__ import annotations
from collections import defaultdict
from itertools import product

from logzero import logger
from typing import Callable
from contexttimer import Timer

from wfomc.cell_graph.cell_graph import build_cell_graphs
from wfomc.network.constraint import PartitionConstraint
from wfomc.utils import MultinomialCoefficients, multinomial_less_than, RingElement, Rational
from wfomc.fol.syntax import Const, Pred, QFFormula


def fast_wfomc(formula: QFFormula,
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


def fast_wfomc_with_pc(formula: QFFormula,
                         domain: set[Const],
                         get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                         partition_constraint: PartitionConstraint) -> RingElement:
    logger.info('Invoke faster WFOMC with partition constraint')
    logger.info('Partition constraint: %s', partition_constraint)
    res = Rational(0, 1)
    domain_size = len(domain)
    for opt_cell_graph, weight in build_cell_graphs(
        formula, get_weight,
        optimized=True, domain_size=domain_size,
        modified_cell_symmetry=True,
        partition_constraint=partition_constraint
    ):
        cliques = opt_cell_graph.cliques
        nonind = opt_cell_graph.nonind
        nonind_map = opt_cell_graph.nonind_map

        pred_partitions: list[list[int]] = list(num for _, num in partition_constraint.partition)
        # partition to cliques
        partition_cliques: dict[int, list[int]] = opt_cell_graph.partition_cliques

        res_ = Rational(0, 1)
        with Timer() as t:
            for configs in product(
                *(list(multinomial_less_than(len(partition_cliques[idx]), constrained_num)) for
                  idx, constrained_num in enumerate(pred_partitions))
            ):
                coef = Rational(1, 1)
                remainings = list()
                # config for the cliques
                overall_config = list(0 for _ in range(len(cliques)))
                # {clique_idx: [number of elements of pred1, pred2, ..., predk]}
                clique_configs = defaultdict(list)
                for idx, (constrained_num, config) in enumerate(zip(pred_partitions, configs)):
                    remainings.append(constrained_num - sum(config))
                    mu = tuple(config) + (constrained_num - sum(config), )
                    coef = coef * MultinomialCoefficients.coef(mu)
                    for num, clique_idx in zip(config, partition_cliques[idx]):
                        overall_config[clique_idx] = overall_config[clique_idx] + num
                        clique_configs[clique_idx].append(num)

                body = opt_cell_graph.get_i1_weight(
                    remainings, overall_config
                )

                for i, clique1 in enumerate(cliques):
                    for j, clique2 in enumerate(cliques):
                        if i in nonind and j in nonind:
                            if i < j:
                                body = body * opt_cell_graph.get_two_table_weight(
                                    (clique1[0], clique2[0])
                                ) ** (overall_config[nonind_map[i]] *
                                      overall_config[nonind_map[j]])

                for l in nonind:
                    body = body * opt_cell_graph.get_J_term(
                        l, tuple(clique_configs[nonind_map[l]])
                    )
                res_ = res_ + coef * body
        res = res + weight * res_
    logger.info('WFOMC time: %s', t.elapsed)

    return res
