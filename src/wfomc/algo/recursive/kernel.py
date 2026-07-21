"""Recursive WFOMC Nauty/isomorphism kernel (framework-owned).

Pure computational core for the recursive algorithm: builds the colored graph,
canonicalizes via pynauty, and runs the isomorphism-cached DFS. Takes plain
cell/edge weights, so it is independent of any WFOMC context.
"""

import logging
import math
from collections import Counter

import pynauty

from wfomc.arithmetic import ArithmeticValue


ENABLE_ISOMORPHISM = True
logger = logging.getLogger(__name__)

# Some pynauty builds ship without the compiled ``Graph``/``certificate``
# extension. When that happens we degrade gracefully: the isomorphism cache key
# falls back to the (color-adjusted) vertex coloring instead of a canonical
# nauty certificate. Results stay correct, but isomorphic states are no longer
# merged, so cache reuse drops. The warning below makes that degradation
# visible instead of silent.
HAS_PYNAUTY_GRAPH = hasattr(pynauty, "Graph") and hasattr(pynauty, "certificate")
_WARNED_NO_PYNAUTY_GRAPH = False


class NautyContext(object):
    def __init__(self, domain_size, cell_weights, edge_weights, arithmetic):
        self.arithmetic = arithmetic
        self.graph = None
        self.layer_num_for_convert = 0
        self.node_num = 0
        self.edge_color_num = 0
        self.edge_weight_to_color = {str(arithmetic.one()): 0}
        self.edge_color_mat = []
        self.vertex_color_no = 0
        self.vertex_weight_to_color = {}
        self.adjacency_dict = {}
        self.cache_for_nauty = {}
        self.ig_cache = IsomorphicGraphCache(domain_size)

        self.edgeWeight_to_edgeColor(edge_weights)
        self.calculate_adjacency_dict(len(cell_weights))
        self.create_graph()

    def edgeWeight_to_edgeColor(self, edge_weights: list[list[ArithmeticValue]]):
        for lst in edge_weights:
            tmp_list = []
            for w in lst:
                if str(w) not in self.edge_weight_to_color:
                    self.edge_color_num += 1
                    self.edge_weight_to_color[str(w)] = self.edge_color_num
                tmp_list.append(self.edge_weight_to_color[str(w)])
            self.edge_color_mat.append(tmp_list)

    def calculate_adjacency_dict(self, cell_num: int):
        """
        The edge weights (colors) is fixed in one problem, so we can calculate the adjacent matrix in advance.
        """
        # convert edge-colored graph to edge-uncolored graph
        # see Section 14: Variations of "Nauty and Traces User’s Guide (Version 2.8.8)" for details
        self.layer_num_for_convert = math.ceil(math.log2(self.edge_color_num + 1))
        self.node_num = cell_num * self.layer_num_for_convert

        adjacency_dict = {}
        for i in range(self.node_num):
            adjacency_dict[i] = []

        c2layers = {}
        for k in range(self.edge_color_num + 1):
            bi = bin(k)
            bi = bi[2:]
            bi = bi[::-1]
            layers = [i for i in range(len(bi)) if bi[i] == "1"]
            c2layers[k] = layers

        for i in range(cell_num):
            for j in range(cell_num):
                layers = c2layers[self.edge_color_mat[i][j]]
                for layer in layers:
                    adjacency_dict[layer * cell_num + i].append(layer * cell_num + j)

        # The vertical threads (each corresponding to one vertex of the original graph)
        # can be connected using either paths or cliques.
        for i in range(cell_num):
            clique = [i + j * cell_num for j in range(self.layer_num_for_convert)]
            for ii in clique:
                for jj in clique:
                    if ii == jj:
                        continue
                    adjacency_dict[ii].append(jj)
        self.adjacency_dict = adjacency_dict

    def create_graph(self):
        if not HAS_PYNAUTY_GRAPH:
            global _WARNED_NO_PYNAUTY_GRAPH
            if not _WARNED_NO_PYNAUTY_GRAPH:
                logger.warning(
                    "pynauty is installed without the compiled Graph/certificate "
                    "extension; RecursiveWFOMC falls back to raw vertex colorings "
                    "as isomorphism cache keys (results stay correct, but cache "
                    "reuse is reduced)."
                )
                _WARNED_NO_PYNAUTY_GRAPH = True
            self.graph = None
            return
        self.graph = pynauty.Graph(
            self.node_num, directed=False, adjacency_dict=self.adjacency_dict
        )

    def update_graph(self, colored_vertices):
        # for speed up, we can modify the function 'set_vertex_coloring' in graph.py of pynauty
        self.graph.set_vertex_coloring(colored_vertices)
        return self.graph

    def get_vertex_color(self, weight):
        if str(weight) not in self.vertex_weight_to_color:
            self.vertex_weight_to_color[str(weight)] = self.vertex_color_no
            self.vertex_color_no += 1
        return self.vertex_weight_to_color[str(weight)]

    def cellWeight_To_vertexColor(self, cell_weights):
        vertex_colors = []
        for w in cell_weights:
            vertex_colors.append(self.get_vertex_color(w))
        color_dict = Counter(vertex_colors)
        # "color_kind" and "color_count" are the keys for IG_CACHE to speed up the search
        # we sort colors in "color_kind" to make sure that the order of colors is fixed
        color_kind = tuple(sorted(color_dict))
        color_count = tuple(color_dict[num] for num in color_kind)
        return vertex_colors, color_kind, color_count

    def extend_vertex_coloring(self, colored_vertices, no_color):
        """
        Extend the vertex set to convert colored edge

        Args:
            colored_vertices: list[int]
                The color no. of vertices.
            no_color: int
                The number of colors.

        Returns:
            vertex_coloring: list[set[int]]
                The color set of vertices.

        Example:
            colored_vertices = [0, 1, 0, 2, 1, 0]
            no_color = 3
            vertex_coloring = extend_vertex_coloring(colored_vertices, no_color)
            print(vertex_coloring)  # [{0, 2, 5}, {1, 4}, {3}]
        """
        # Extend the vertex set to convert colored edge
        ext_colored_vertices = []
        for i in range(self.layer_num_for_convert):
            ext_colored_vertices += [x + no_color * i for x in colored_vertices]

        # Get color set of vertices
        no_color *= self.layer_num_for_convert
        vertex_coloring = [set() for _ in range(no_color)]
        for i in range(len(ext_colored_vertices)):
            c = ext_colored_vertices[i]
            vertex_coloring[c].add(i)
        return vertex_coloring


class IsomorphicGraphCache(object):
    def __init__(self, domain_size: int):
        self.cache = []
        self.cache_hit_count = []
        for _ in range(domain_size):
            self.cache.append({})
            self.cache_hit_count.append(0)

    def get(
        self, level: int, color_kind: tuple[int], color_count: tuple[int], can_label
    ):
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

    def set(
        self,
        level: int,
        color_kind: tuple[int],
        color_count: tuple[int],
        can_label,
        value,
    ):
        if color_kind not in self.cache[level]:
            self.cache[level][color_kind] = {}
        if color_count not in self.cache[level][color_kind]:
            self.cache[level][color_kind][color_count] = {}
        self.cache[level][color_kind][color_count][can_label] = value


def adjust_vertex_coloring(colored_vertices):
    """
    Adjust the color no. of vertices to make the color no. start from 0 and be continuous.
    Args:
        colored_vertices: list[int]
            The color no. of vertices.

        Returns:
            new_colored_vertices: list[int]
                The adjusted color no. of vertices.
            num_color: int
                The number of colors.
    Example:
        colored_vertices = [7, 5, 7, 3, 5, 7]
        new_colored_vertices, num_color = adjust_vertex_coloring(colored_vertices)
        print(new_colored_vertices)  # [2, 1, 2, 0, 1, 2]
        print(num_color)  # 3
    """
    sorted_colors = sorted(set(colored_vertices))
    rank = {v: i for i, v in enumerate(sorted_colors)}
    return [rank[x] for x in colored_vertices], len(sorted_colors)


def dfs_wfomc_real(
    cell_weights,
    edge_weights,
    domain_size,
    nauty_ctx: NautyContext,
):
    res = nauty_ctx.arithmetic.zero()
    cell_num = len(cell_weights)
    for cell_idx in range(cell_num):
        cell_weight = cell_weights[cell_idx]
        new_cell_weights = [
            nauty_ctx.arithmetic.multiply(
                cell_weights[i],
                edge_weights[cell_idx][i],
            )
            for i in range(cell_num)
        ]
        if domain_size - 1 == 1:
            value = nauty_ctx.arithmetic.sum(new_cell_weights)
        else:
            # convert cell weights to vertex colors
            original_vertex_colors, vertex_color_kind, vertex_color_count = (
                nauty_ctx.cellWeight_To_vertexColor(new_cell_weights)
            )
            if ENABLE_ISOMORPHISM:
                # adjust the color no. of vertices to make them start from 0 and be continuous to add the hit rate of "CACHE_FOR_NAUTY"
                # here we dont need to consider the different "original_vertex_colors"s with the same "adjust_vertex_colors",
                # since our IG_CACHE has multiple keys (vertex_color_kind, vertex_color_count) to distinguish them
                # even if they have the same "adjust_vertex_colors" but different "vertex_color_kind" or "vertex_color_count"
                adjust_vertex_colors, no_color = adjust_vertex_coloring(
                    original_vertex_colors
                )
                if not HAS_PYNAUTY_GRAPH:
                    can_label = tuple(adjust_vertex_colors)
                elif tuple(adjust_vertex_colors) not in nauty_ctx.cache_for_nauty:
                    can_label = pynauty.certificate(
                        nauty_ctx.update_graph(
                            nauty_ctx.extend_vertex_coloring(
                                adjust_vertex_colors, no_color
                            )
                        )
                    )
                    nauty_ctx.cache_for_nauty[tuple(adjust_vertex_colors)] = can_label
                else:
                    can_label = nauty_ctx.cache_for_nauty[tuple(adjust_vertex_colors)]
            else:
                can_label = tuple(original_vertex_colors)

            value = nauty_ctx.ig_cache.get(
                domain_size - 1, vertex_color_kind, vertex_color_count, can_label
            )
            if value is None:
                value = dfs_wfomc_real(
                    new_cell_weights,
                    edge_weights,
                    domain_size - 1,
                    nauty_ctx,
                )
                nauty_ctx.ig_cache.set(
                    domain_size - 1,
                    vertex_color_kind,
                    vertex_color_count,
                    can_label,
                    value,
                )
        res = nauty_ctx.arithmetic.add(
            res,
            nauty_ctx.arithmetic.multiply(cell_weight, value),
        )
    return res


__all__ = [
    "HAS_PYNAUTY_GRAPH",
    "ENABLE_ISOMORPHISM",
    "NautyContext",
    "IsomorphicGraphCache",
    "adjust_vertex_coloring",
    "dfs_wfomc_real",
]
