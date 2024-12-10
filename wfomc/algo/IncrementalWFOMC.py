from typing import Callable
from functools import reduce

from wfomc.cell_graph import build_cell_graphs
from wfomc.context.wfomc_context import WFOMCContext
from wfomc.utils import RingElement, Rational
from wfomc.fol.syntax import Const, Pred, QFFormula


def incremental_wfomc(context: WFOMCContext) -> RingElement:
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    leq_pred = context.leq_pred
    predecessor_pred = context.predecessor_pred
    res = Rational(0, 1)
    domain_size = len(domain)
    for cell_graph, weight in build_cell_graphs(
        formula, get_weight,
        leq_pred=leq_pred,
        predecessor_pred=predecessor_pred
    ):
        # cell_graph.show()
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        domain_size = len(domain)

        if predecessor_pred is None:
            table = dict(
                (
                    tuple(int(k == i) for k in range(n_cells)),
                    cell_graph.get_cell_weight(cell),
                )
                for i, cell in enumerate(cells)
            )
        else:
            table = dict(
                (
                    (
                        tuple(int(k == i) for k in range(n_cells)),
                        cell
                    ),
                    cell_graph.get_cell_weight(cell),
                )
                for i, cell in enumerate(cells)
            )
        for _ in range(domain_size - 1):
            old_table = table
            table = dict()
            for j, cell in enumerate(cells):
                w = cell_graph.get_cell_weight(cell)
                for key, w_old in old_table.items():
                    if predecessor_pred is None:
                        ivec = key
                        w_new = w_old * w * reduce(
                            lambda x, y: x * y,
                            (
                                cell_graph.get_two_table_weight((cell, cells[k]))
                                ** int(ivec[k]) for k in range(n_cells)
                            ),
                            Rational(1, 1)
                        )
                    else:
                        ivec, last_cell = key
                        w_new = w_old * w
                        for k, other_cell in enumerate(cells):
                            if other_cell == last_cell:
                                w_new = (
                                    w_new * cell_graph.get_two_table_with_pred_weight((cell, other_cell))
                                    * cell_graph.get_two_table_weight((cell, other_cell)) ** max(ivec[k] - 1, 0)
                                )
                            else:
                                w_new = w_new * cell_graph.get_two_table_weight((cell, other_cell)) ** ivec[k]
                    ivec = list(ivec)
                    ivec[j] += 1
                    ivec = tuple(ivec)
                    if predecessor_pred is None:
                        w_new = w_new + table.get(ivec, Rational(0, 1))
                        table[ivec] = w_new
                    else:
                        w_new = w_new + table.get((ivec, cell), Rational(0, 1))
                        table[(tuple(ivec), cell)] = w_new
        res = res + weight * sum(table.values())

    # if leq_pred is not None:
    #     res *= Rational(math.factorial(domain_size), 1)
    return res
