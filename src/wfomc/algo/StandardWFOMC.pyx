# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from wfomc.cell_graph import CellGraph, Cell, build_cell_graphs
from wfomc.context import WFOMCContext
from wfomc.utils import MultinomialCoefficients, multinomial, RingElement, Rational


def get_config_weight_standard(cell_graph: CellGraph,
                               cell_config: dict) -> object:
    cdef int i, j
    cdef object res, n_i, n_j
    res = Rational(1, 1)
    cells_items = list(cell_config.items())
    for i, (cell_i, n_i) in enumerate(cells_items):
        if n_i == 0:
            continue
        res = res * cell_graph.get_nullary_weight(cell_i)
        break
    for i, (cell_i, n_i) in enumerate(cells_items):
        if n_i == 0:
            continue
        res = res * cell_graph.get_cell_weight(cell_i) ** n_i
        res = res * cell_graph.get_two_table_weight(
            (cell_i, cell_i)
        ) ** (n_i * (n_i - 1) // 2)
        for j, (cell_j, n_j) in enumerate(cells_items):
            if j <= i:
                continue
            if n_j == 0:
                continue
            res = res * cell_graph.get_two_table_weight(
                (cell_i, cell_j)
            ) ** (n_i * n_j)
    return res


def standard_wfomc(context: WFOMCContext) -> object:
    cdef object res, coef, weight
    cdef int domain_size = len(context.domain)
    cdef int n_cells
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    res = Rational(0, 1)

    for cell_graph, weight in build_cell_graphs(formula, get_weight):
        res_ = Rational(0, 1)
        cells = cell_graph.get_cells()
        n_cells = len(cells)
        for partition in multinomial(n_cells, domain_size):
            coef = MultinomialCoefficients.coef(partition)
            cell_config = dict(zip(cells, partition))
            res_ = res_ + coef * get_config_weight_standard(cell_graph, cell_config)
        res = res + weight * res_
    return res
