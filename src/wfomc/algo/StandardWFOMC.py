from flint import fmpq as Rational

from wfomc.cell_graph import CellGraph, Cell
from wfomc.context import CellEvidenceAllocation, WFOMCContext
from wfomc.utils import RingElement

def get_config_weight_standard(cell_graph: CellGraph,
                               cell_config: dict[Cell, int]) -> RingElement:
    res = 1
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
    domain = context.domain
    res = Rational(0, 1)
    domain_size = len(domain)
    for cell_graph, weight in context.build_cell_graphs():
        res_ = Rational(0, 1)
        cells = cell_graph.get_cells()
        allocation = context.cell_evidence_allocation(cells)
        if allocation is None:
            allocation = CellEvidenceAllocation.unconstrained(
                len(cells), domain_size
            )
        for config, coef in allocation.iter_config_coefficients():
            cell_config = dict(zip(cells, config))
            res_ = res_ + coef * get_config_weight_standard(
                cell_graph, cell_config
            )
        res = res + weight * res_
    return res
