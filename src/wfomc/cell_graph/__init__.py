from wfomc._compat import try_import_cython as _try_cy

_cg_mod = _try_cy("wfomc.cell_graph.cell_graph", "wfomc.cell_graph.cell_graph")
CellGraph = _cg_mod.CellGraph
OptimizedCellGraph = _cg_mod.OptimizedCellGraph
OptimizedCellGraphWithPC = _cg_mod.OptimizedCellGraphWithPC
build_cell_graphs = _cg_mod.build_cell_graphs

from .components import Cell, TwoTable

__all__ = [
    'CellGraph',
    'OptimizedCellGraph',
    'OptimizedCellGraphWithPC',
    'build_cell_graphs',
    'Cell',
    'TwoTable',
]
