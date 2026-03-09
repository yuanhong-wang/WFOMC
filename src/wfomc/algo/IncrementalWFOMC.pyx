# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from wfomc.cell_graph import build_cell_graphs
from wfomc.context import WFOMCContext
from wfomc.network import UnaryEvidenceEncoding
from wfomc.utils import RingElement, Rational, MultinomialCoefficients


def _helper(cell, pc_pred, pc_ccs):
    """Find first partition pred index that cell satisfies and has remaining count."""
    for i, p in enumerate(pc_pred):
        if cell.is_positive(p) and pc_ccs[i] > 0:
            return i
    return None


def _make_ivec(n_cells, selected_idx):
    """Build an n_cells-length tuple with 1 at selected_idx, 0 elsewhere."""
    result = []
    for k in range(n_cells):
        result.append(1 if k == selected_idx else 0)
    return tuple(result)


def _update_ccs(ccs, selected_idx):
    """Build new ccs tuple with ccs[selected_idx] decremented by 1."""
    result = []
    for k, cc in enumerate(ccs):
        result.append(cc - 1 if k == selected_idx else cc)
    return tuple(result)


def incremental_wfomc(context: WFOMCContext, circle_len=None) -> object:
    cdef int pred_max_order = 0
    cdef int domain_size = len(context.domain)
    cdef int circle_len_int
    cdef int cur_idx, j_idx, ki, pred_idx, n_cells
    cdef object accum

    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    leq_pred = context.leq_pred
    res = Rational(0, 1)
    predecessor_preds = context.predecessor_preds
    pred_orders = None
    if predecessor_preds is not None:
        pred_orders = list(predecessor_preds.keys())
        pred_max_order = max(pred_orders)
    circular_predecessor_pred = context.circular_predecessor_pred
    if circle_len is None:
        circle_len_int = domain_size
    else:
        circle_len_int = circle_len

    for cell_graph, weight in build_cell_graphs(
        formula, get_weight,
        leq_pred=leq_pred,
        predecessor_preds=predecessor_preds
    ):
        cells = cell_graph.get_cells()
        n_cells = len(cells)

        if context.unary_evidence_encoding == UnaryEvidenceEncoding.PC:
            pc_pred, pc_ccs = zip(*context.partition_constraint.partition)
            table = dict()
            for i_idx, cell in enumerate(cells):
                j = _helper(cell, pc_pred, pc_ccs)
                if j is None:
                    continue
                key = (
                    _make_ivec(n_cells, i_idx),
                    None if pred_orders is None else tuple(
                        cell for _ in range(pred_max_order)
                    ),
                    None if circular_predecessor_pred is None else cell
                )
                table[key] = (
                    cell_graph.get_cell_weight(cell),
                    _update_ccs(pc_ccs, j)
                )
        else:
            table = dict()
            for i_idx, cell in enumerate(cells):
                key = (
                    _make_ivec(n_cells, i_idx),
                    None if pred_orders is None else tuple(
                        cell for _ in range(pred_max_order)
                    ),
                    None if circular_predecessor_pred is None else cell
                )
                table[key] = (
                    cell_graph.get_cell_weight(cell),
                    None
                )

        for cur_idx in range(domain_size - 1):
            old_table = table
            table = dict()
            for j_idx, cell in enumerate(cells):
                w = cell_graph.get_cell_weight(cell)
                for (ivec, last_cells, first_cell), (w_old, old_ccs) in old_table.items():
                    old_ivec = list(ivec)
                    if old_ccs is not None:
                        idx = _helper(cell, pc_pred, old_ccs)
                        if idx is None:
                            continue
                        new_ccs = _update_ccs(old_ccs, idx)
                    else:
                        new_ccs = None

                    w_new = w_old * w
                    if cur_idx == circle_len_int - 2 and first_cell is not None:
                        w_new = w_new * cell_graph.get_two_table_with_pred_weight(
                            (first_cell, cell), 1
                        )
                        old_ivec[cells.index(first_cell)] -= 1
                    if last_cells is not None:
                        for pred_idx in pred_orders:
                            if cur_idx >= pred_idx - 1:
                                pred_cell = last_cells[pred_max_order - pred_idx]
                                w_new = w_new * cell_graph.get_two_table_with_pred_weight(
                                    (cell, pred_cell), pred_idx
                                )
                                old_ivec[cells.index(pred_cell)] -= 1
                        new_last_cells = last_cells[1:] + (cell,)
                    else:
                        new_last_cells = None

                    accum = Rational(1, 1)
                    for ki in range(n_cells):
                        accum = accum * cell_graph.get_two_table_weight(
                            (cell, cells[ki])
                        ) ** old_ivec[ki]
                    w_new = w_new * accum

                    new_ivec = list(ivec)
                    new_ivec[j_idx] += 1
                    ivec = tuple(new_ivec)
                    new_last_cells = (
                        tuple(new_last_cells)
                        if new_last_cells is not None else None
                    )
                    w_new = w_new + table.get(
                        (ivec, new_last_cells, first_cell),
                        (Rational(0, 1), ())
                    )[0]
                    table[(tuple(ivec), new_last_cells, first_cell)] = (
                        w_new, new_ccs
                    )
        res = res + weight * sum(w for w, _ in table.values())

    if context.unary_evidence_encoding == UnaryEvidenceEncoding.PC:
        res = res / MultinomialCoefficients.coef(
            tuple(
                v for _, v in context.partition_constraint.partition
            )
        )
    return res
