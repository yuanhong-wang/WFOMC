from functools import reduce

from flint import fmpq as Rational

from wfomc.context import CellEvidenceAllocation, WFOMCContext
from wfomc.utils import RingElement


def incremental_wfomc(context: WFOMCContext,
                      circle_len: int = None) -> RingElement:
    domain = context.domain
    leq_pred = context.leq_pred
    res = Rational(0, 1)
    predecessor_preds = context.predecessor_preds
    pred_orders = None
    pred_max_order = 0
    if predecessor_preds is not None:
        pred_orders = list(predecessor_preds.keys())
        pred_max_order = max(pred_orders)
    circular_predecessor_pred = context.circular_predecessor_pred
    domain_size = len(domain)
    if circle_len is None:
        circle_len = domain_size

    for cell_graph, weight in context.build_cell_graphs(
        leq_pred=leq_pred,
        predecessor_preds=predecessor_preds
    ):
        # cell_graph.show()
        cells = cell_graph.get_cells()
        n_cells = len(cells)

        def step_weight(active_cells, cell, j, ivec, last_cells, first_cell,
                        w_old, cur_idx):
            """Weight of extending a partial run by placing ``cell`` at the next
            position. ``active_cells`` indexes ``ivec`` (the full cell list, or
            the evidence-compatible subset for the threaded path). Returns
            (w_new, new_ivec, new_last_cells)."""
            old_ivec = list(ivec)
            w_new = w_old * cell_graph.get_cell_weight(cell)
            # circular predecessor: close the cycle on the last step
            # NOTE: only support either circular predecessor or predecessors
            if cur_idx == circle_len - 2 and first_cell is not None:
                w_new = w_new * cell_graph.get_two_table_with_pred_weight(
                    (first_cell, cell), 1
                )
                old_ivec[active_cells.index(first_cell)] -= 1
            # predecessors
            if last_cells is not None:
                for pred_idx in pred_orders:
                    if cur_idx >= pred_idx - 1:
                        pred_cell = last_cells[-pred_idx]
                        w_new = w_new * cell_graph.get_two_table_with_pred_weight(
                            (cell, pred_cell), pred_idx
                        )
                        old_ivec[active_cells.index(pred_cell)] -= 1
                new_last_cells = last_cells[1:] + (cell,)
            else:
                new_last_cells = None
            w_new = w_new * reduce(
                lambda x, y: x * y,
                (
                    cell_graph.get_two_table_weight((cell, other_cell)) ** old_ivec[k]
                    for k, other_cell in enumerate(active_cells)
                )
            )
            new_ivec = tuple(
                (num if k != j else num + 1) for k, num in enumerate(ivec)
            )
            new_last_cells = (
                tuple(new_last_cells) if new_last_cells is not None else None
            )
            return w_new, new_ivec, new_last_cells

        def init_state(cell):
            return (
                None if pred_orders is None else tuple(
                    cell for _ in range(pred_max_order)
                ),
                None if circular_predecessor_pred is None else cell,
            )

        allocation = context.cell_evidence_allocation(cells)
        if allocation is not None:
            graph_res = _threaded_graph(
                cell_graph, cells, allocation, domain_size,
                step_weight, init_state,
            )
            res = res + weight * graph_res
            continue

        # Non-threaded path: no unary evidence.
        table = dict()
        for i, cell in enumerate(cells):
            last_cells, first_cell = init_state(cell)
            table[
                (
                    tuple(int(k == i) for k in range(n_cells)),
                    last_cells,
                    first_cell,
                )
            ] = cell_graph.get_cell_weight(cell)

        for cur_idx in range(domain_size - 1):
            old_table = table
            table = dict()
            for j, cell in enumerate(cells):
                for (ivec, last_cells, first_cell), w_old in old_table.items():
                    w_new, new_ivec, new_last_cells = step_weight(
                        cells, cell, j, ivec, last_cells, first_cell,
                        w_old, cur_idx
                    )
                    key = (new_ivec, new_last_cells, first_cell)
                    table[key] = table.get(key, Rational(0, 1)) + w_new

        graph_res = Rational(0, 1)
        for (ivec, _, _), w in table.items():
            graph_res += w
        res = res + weight * graph_res

    return res


def _threaded_graph(
    cell_graph,
    cells,
    allocation: CellEvidenceAllocation,
    domain_size,
    step_weight,
    init_state,
) -> RingElement:
    """Compute a cell-graph contribution by threading evidence-profile capacities.

    The DP state carries the allocation's remaining evidence-profile capacities.
    Summing over valid evidence-choice paths counts ordered assignments, which
    are removed by ``normalize_ordered_evidence_assignments`` before returning.
    """
    active_indices = allocation.compatible_cell_indices
    cells = [cells[idx] for idx in active_indices]
    n_cells = len(cells)
    initial_remaining = allocation.initial_remaining_counts()

    # Seed: place the first element.
    table: dict = dict()
    for i, (cell, original_idx) in enumerate(zip(cells, active_indices)):
        last_cells, first_cell = init_state(cell)
        ivec0 = tuple(int(k == i) for k in range(n_cells))
        w0 = cell_graph.get_cell_weight(cell)
        for rem in allocation.next_remaining_counts(
            initial_remaining, original_idx
        ):
            key = (ivec0, last_cells, first_cell, rem)
            table[key] = table.get(key, Rational(0, 1)) + w0

    for cur_idx in range(domain_size - 1):
        old_table = table
        table = dict()
        for j, (cell, original_idx) in enumerate(zip(cells, active_indices)):
            for (
                ivec,
                last_cells,
                first_cell,
                old_rem,
            ), w_old in old_table.items():
                new_remaining_states = allocation.next_remaining_counts(
                    old_rem, original_idx
                )
                if not new_remaining_states:
                    continue
                w_new, new_ivec, new_last_cells = step_weight(
                    cells, cell, j, ivec, last_cells, first_cell,
                    w_old, cur_idx
                )
                for new_rem in new_remaining_states:
                    key = (new_ivec, new_last_cells, first_cell, new_rem)
                    table[key] = table.get(key, Rational(0, 1)) + w_new

    zero_rem = tuple(0 for _ in allocation.evidence_profile_sizes)
    graph_res = Rational(0, 1)
    for (_ivec, _last, _first, rem), w in table.items():
        if rem == zero_rem:
            graph_res += w
    return allocation.normalize_ordered_evidence_assignments(graph_res)
