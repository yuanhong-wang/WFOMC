from collections import defaultdict
from itertools import product
from loguru import logger
from contexttimer import Timer

from flint import fmpq as Rational

from wfomc.context import WFOMCContext
from wfomc.utils import MultinomialCoefficients, multinomial_less_than, RingElement


def fast_wfomc(context: WFOMCContext,
               modified_cell_symmetry: bool = False) -> RingElement:
    if context.uses_lifted_unary_evidence:
        return _fast_wfomc_with_evidence(context)

    return _fast_wfomc(context, modified_cell_symmetry)


def _fast_wfomc(
    context: WFOMCContext,
    modified_cell_symmetry: bool = False,
) -> RingElement:
    domain_size = len(context.domain)
    res = Rational(0, 1)
    for opt_cell_graph, weight in context.build_cell_graphs(
        optimized=True,
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
    logger.info('WFOMC time: {}', t.elapsed)
    return res


def _fast_wfomc_with_evidence(context: WFOMCContext) -> RingElement:
    logger.info('Invoke faster WFOMC with cell evidence allocation')
    res = Rational(0, 1)
    domain_size = len(context.domain)
    for opt_cell_graph, weight in context.build_cell_graphs(
        optimized=True,
        domain_size=domain_size,
        modified_cell_symmetry=True,
    ):
        cliques = opt_cell_graph.cliques
        nonind = opt_cell_graph.nonind
        nonind_map = opt_cell_graph.nonind_map
        evidence_profile_sizes = opt_cell_graph.evidence_profile_sizes
        evidence_profile_cliques = opt_cell_graph.evidence_profile_cliques

        res_ = Rational(0, 1)
        with Timer() as t:
            for configs in product(
                *(
                    list(multinomial_less_than(
                        len(evidence_profile_cliques[evidence_profile_idx]),
                        constrained_num,
                    ))
                    for evidence_profile_idx, constrained_num in enumerate(
                        evidence_profile_sizes
                    )
                )
            ):
                coef = Rational(1, 1)
                remainings = []
                overall_config = [0 for _ in range(len(cliques))]
                clique_configs = defaultdict(list)
                for evidence_profile_idx, (constrained_num, config) in enumerate(
                    zip(evidence_profile_sizes, configs)
                ):
                    remainings.append(constrained_num - sum(config))
                    mu = tuple(config) + (constrained_num - sum(config),)
                    coef *= MultinomialCoefficients.coef(mu)
                    for num, clique_idx in zip(
                        config, evidence_profile_cliques[evidence_profile_idx]
                    ):
                        overall_config[clique_idx] += num
                        clique_configs[clique_idx].append(num)

                body = opt_cell_graph.get_i1_weight(
                    remainings, overall_config
                )

                for i, clique1 in enumerate(cliques):
                    for j, clique2 in enumerate(cliques):
                        if i in nonind and j in nonind and i < j:
                            body *= opt_cell_graph.get_two_table_weight(
                                (clique1[0], clique2[0])
                            ) ** (
                                overall_config[nonind_map[i]]
                                * overall_config[nonind_map[j]]
                            )

                for clique_idx in nonind:
                    body *= opt_cell_graph.get_J_term(
                        clique_idx,
                        tuple(clique_configs[nonind_map[clique_idx]]),
                    )
                res_ += coef * body
        res += weight * res_
    logger.info('WFOMC time: {}', t.elapsed)
    return res
