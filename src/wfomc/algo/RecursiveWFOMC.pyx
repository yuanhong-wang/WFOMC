# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import math
import functools
from collections import Counter
import pynauty
import networkx as nx

from wfomc.cell_graph import CellGraph, build_cell_graphs
from wfomc.utils import RingElement, Rational
from wfomc.utils.polynomial_flint import expand
from wfomc.fol.syntax import Const, Pred, QFFormula
from wfomc.context import WFOMCContext


class NautyContext(object):
    def __init__(self, domain_size, cell_weights, edge_weights):
        self.graph = None
        self.layer_num_for_convert = 0
        self.node_num = 0
        self.edge_color_num = 0
        self.edge_weight_to_color = {1: 0}
        self.edge_color_mat = []
        self.vertex_color_no = 0
        self.vertex_weight_to_color = {}
        self.adjacency_dict = {}
        self.cache_for_nauty = {}
        self.ig_cache = IsomorphicGraphCache(domain_size)

        self.edgeWeight_to_edgeColor(edge_weights)
        self.calculate_adjacency_dict(len(cell_weights))
        self.create_graph()

    def edgeWeight_to_edgeColor(self, edge_weights):
        for lst in edge_weights:
            tmp_list = []
            for w in lst:
                if str(w) not in self.edge_weight_to_color:
                    self.edge_color_num += 1
                    self.edge_weight_to_color[str(w)] = self.edge_color_num
                tmp_list.append(self.edge_weight_to_color[str(w)])
            self.edge_color_mat.append(tmp_list)

    def calculate_adjacency_dict(self, int cell_num):
        cdef int i, j, l, k
        self.layer_num_for_convert = math.ceil(math.log2(self.edge_color_num + 1))
        self.node_num = cell_num * self.layer_num_for_convert

        adjacency_dict = {}
        for i in range(self.node_num):
            adjacency_dict[i] = []

        c2layers = {}
        for k in range(self.edge_color_num + 1):
            bi = bin(k)[2:][::-1]
            layers = [i for i in range(len(bi)) if bi[i] == '1']
            c2layers[k] = layers

        for i in range(cell_num):
            for j in range(cell_num):
                layers = c2layers[self.edge_color_mat[i][j]]
                for l in layers:
                    adjacency_dict[l * cell_num + i].append(l * cell_num + j)

        for i in range(cell_num):
            clique = [i + j * cell_num for j in range(self.layer_num_for_convert)]
            for ii in clique:
                for jj in clique:
                    if ii != jj:
                        adjacency_dict[ii].append(jj)
        self.adjacency_dict = adjacency_dict

    def create_graph(self):
        self.graph = pynauty.Graph(
            self.node_num, directed=False,
            adjacency_dict=self.adjacency_dict
        )

    def update_graph(self, colored_vertices):
        self.graph.set_vertex_coloring(colored_vertices)
        return self.graph

    def get_vertex_color(self, weight):
        if str(weight) not in self.vertex_weight_to_color:
            self.vertex_weight_to_color[str(weight)] = self.vertex_color_no
            self.vertex_color_no += 1
        return self.vertex_weight_to_color[str(weight)]

    def cellWeight_To_vertexColor(self, cell_weights):
        vertex_colors = [self.get_vertex_color(w) for w in cell_weights]
        color_dict = Counter(vertex_colors)
        color_kind = tuple(sorted(color_dict))
        color_count = tuple(color_dict[num] for num in color_kind)
        return vertex_colors, color_kind, color_count

    def extend_vertex_coloring(self, colored_vertices, int no_color):
        cdef int i
        ext_colored_vertices = []
        for i in range(self.layer_num_for_convert):
            ext_colored_vertices += [x + no_color * i for x in colored_vertices]
        no_color *= self.layer_num_for_convert
        vertex_coloring = [set() for _ in range(no_color)]
        for i in range(len(ext_colored_vertices)):
            vertex_coloring[ext_colored_vertices[i]].add(i)
        return vertex_coloring


class TreeNode(object):
    def __init__(self, cell_weights, depth):
        self.cell_weights = cell_weights
        self.depth = depth
        self.cell_to_children = dict()


class IsomorphicGraphCache(object):
    def __init__(self, int domain_size):
        self.cache = [{} for _ in range(domain_size)]
        self.cache_hit_count = [0] * domain_size

    def get(self, int level, color_kind, color_count, can_label):
        if color_kind not in self.cache[level]:
            self.cache[level][color_kind] = {}
            return None
        if color_count not in self.cache[level][color_kind]:
            self.cache[level][color_kind][color_count] = {}
            return None
        if can_label not in self.cache[level][color_kind][color_count]:
            return None
        self.cache_hit_count[level] += 1
        return self.cache[level][color_kind][color_count][can_label]

    def set(self, int level, color_kind, color_count, can_label, value):
        if color_kind not in self.cache[level]:
            self.cache[level][color_kind] = {}
        if color_count not in self.cache[level][color_kind]:
            self.cache[level][color_kind][color_count] = {}
        self.cache[level][color_kind][color_count][can_label] = value


def adjust_vertex_coloring(colored_vertices):
    sorted_colors = sorted(set(colored_vertices))
    rank = {v: i for i, v in enumerate(sorted_colors)}
    return [rank[x] for x in colored_vertices], len(sorted_colors)


ENABLE_ISOMORPHISM = True

def dfs_wfomc_real(cell_weights, edge_weights, int domain_size,
                   nauty_ctx: NautyContext, node=None) -> object:
    cdef int l, cell_num = len(cell_weights)
    cdef object res, w_l, value
    res = 0

    for l in range(cell_num):
        w_l = cell_weights[l]
        new_cell_weights = [expand(cell_weights[i] * edge_weights[l][i])
                            for i in range(cell_num)]
        if domain_size - 1 == 1:
            value = sum(new_cell_weights)
        else:
            original_vertex_colors, vertex_color_kind, vertex_color_count = \
                nauty_ctx.cellWeight_To_vertexColor(new_cell_weights)
            if ENABLE_ISOMORPHISM:
                adjust_vertex_colors, no_color = adjust_vertex_coloring(
                    original_vertex_colors
                )
                key = tuple(adjust_vertex_colors)
                if key not in nauty_ctx.cache_for_nauty:
                    can_label = pynauty.certificate(
                        nauty_ctx.update_graph(
                            nauty_ctx.extend_vertex_coloring(
                                adjust_vertex_colors, no_color
                            )
                        )
                    )
                    nauty_ctx.cache_for_nauty[key] = can_label
                else:
                    can_label = nauty_ctx.cache_for_nauty[key]
            else:
                can_label = tuple(original_vertex_colors)

            value = nauty_ctx.ig_cache.get(
                domain_size - 1, vertex_color_kind, vertex_color_count, can_label
            )
            if value is None:
                value = dfs_wfomc_real(
                    new_cell_weights, edge_weights, domain_size - 1, nauty_ctx, None
                )
                value = expand(value)
                nauty_ctx.ig_cache.set(
                    domain_size - 1, vertex_color_kind, vertex_color_count,
                    can_label, value
                )
        res = res + w_l * value
    return res


def find_independent_sets(cell_graph: CellGraph):
    cdef int i, j
    g = nx.Graph()
    cells = cell_graph.cells
    g.add_nodes_from(range(len(cells)))
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if cell_graph.get_two_table_weight(
                    (cells[i], cells[j])) != Rational(1, 1):
                g.add_edge(i, j)

    self_loop = set()
    for i in range(len(cells)):
        if cell_graph.get_two_table_weight(
                (cells[i], cells[i])) != Rational(1, 1):
            self_loop.add(i)

    non_self_loop = g.nodes - self_loop
    if len(non_self_loop) == 0:
        i1_ind = set()
    else:
        i1_ind = set(nx.maximal_independent_set(g.subgraph(non_self_loop)))
    g_ind = set(nx.maximal_independent_set(g, nodes=i1_ind))
    i2_ind = g_ind.difference(i1_ind)
    non_ind = g.nodes - i1_ind - i2_ind
    return list(i1_ind), list(i2_ind), list(non_ind)


ROOT = TreeNode([], 0)


def recursive_wfomc(context: WFOMCContext) -> object:
    cdef int domain_size = len(context.domain)
    cdef object res, weight, res_
    formula = context.formula
    domain = context.domain
    get_weight = context.get_weight
    leq_pred = context.leq_pred
    res = Rational(0, 1)

    for cell_graph, weight in build_cell_graphs(
            formula, get_weight, leq_pred=leq_pred):
        cell_weights = cell_graph.get_all_weights()[0]
        edge_weights = cell_graph.get_all_weights()[1]

        nauty_ctx = NautyContext(domain_size, cell_weights, edge_weights)
        global ROOT
        ROOT.cell_weights = cell_weights
        res_ = dfs_wfomc_real(cell_weights, edge_weights, domain_size, nauty_ctx, ROOT)
        res = res + weight * res_
    return res
