from __future__ import annotations
from typing import Callable

from wfomc.cell_graph.cell_graph import CellGraph, Cell, build_cell_graphs
from wfomc.context.wfomc_context import WFOMCContext
from wfomc.utils import MultinomialCoefficients, multinomial, RingElement, Rational
from wfomc.fol.syntax import Const, Pred, QFFormula

def get_config_weight_standard(cell_graph: CellGraph,
                               cell_config: dict[Cell, int]) -> RingElement:
    res = Rational(1, 1)
    for cell, n in cell_config.items():
        if n > 0:
            # NOTE: nullary weight is multiplied once
            res = res * cell_graph.get_nullary_weight(cell)
            break
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


def standard_wfomc(context: WFOMCContext) -> RingElement:
    # cell_graph.show()
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
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
