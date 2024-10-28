import math
from collections import Counter
from typing import Callable
import pynauty

from sympy import factorint, factor_list
from wfomc.cell_graph import CellGraph, build_cell_graphs
from wfomc.utils import RingElement, Rational
from wfomc.utils.polynomial import expand
from wfomc.fol.syntax import Const, Pred, QFFormula

class TreeNode(object):
    def __init__(self, cell_weights, depth):
        self.cell_weights = cell_weights
        self.depth = depth
        self.cell_to_children = dict[int, TreeNode]()

def print_tree(node: TreeNode):
    print("  " * node.depth, node.depth, " ", node.cell_weights)
    for k,v in node.cell_to_children.items():
        print_tree(v)

class IsomorphicGraphCache(object):
    def init(self, domain_size: int):
        self.cache = []
        self.cache_hit_count = []
        for _ in range(domain_size):
            self.cache.append({})
            self.cache_hit_count.append(0)
    
    def get(self, level: int, color_kind: tuple[int], color_count: tuple[int], can_label):
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
    
    def set(self, level: int, color_kind: tuple[int], color_count: tuple[int], can_label, value):
        if color_kind not in self.cache[level]:
            self.cache[level][color_kind] = {}
        if color_count not in self.cache[level][color_kind]:
            self.cache[level][color_kind][color_count] = {}
        self.cache[level][color_kind][color_count][can_label] = value

# only for Alog 'dfs_wfomc_real'
PRINT_TREE = False
ROOT = TreeNode([], 0)
# cache for isormophic graph
IG_CACHE = IsomorphicGraphCache()
# weight adjacent matrix of original cell graph
ORI_WEIGHT_ADJ_MAT = []
# how many layers is needed when convert edge-colored graph
LAYERS_NUM_FOR_CONVERT = 0
# the number of colors of edge
EDGE_COLOR_NUM = 0
# key: edge weight, value: edge color
EDGE_WEIGHT2COLOR_MAP = {1:0}
# edge color matrix of original cell graph (ORI_WEIGHT_ADJ_MAT + EDGE_WEIGHT2COLOR_MAP = COLOR_ADJ_MAT)
COLOR_ADJ_MAT = []
# the global no. of color of vertex
VERTEX_COLOR_NO = 0
# the number of vertries of the original cell graph
CELLS_NUM = 0
# the number of vertries of the extend cell graph
EXT_CELLS_NUM = 0
# key: vertex weight, value: vertex color
VERTEX_WEIGHT2COLOR_MAP: dict[any, int] = {}
# adjacency_dict
ADJACENCY_DICT = {}
# reduce the call of pynauty.certificate
CACHE_FOR_NAUTY = {}
# cache the isomorphic graph
ENABLE_ISOMORPHISM = True
# key is factor and value is the index of factor
FACTOR2INDEX_MAP = {}
# the index of factor 0
ZERO_FACTOR_INDEX = -1
# the factor of edge weights
FACTOR_ADJ_MAT = []

def update_factor_dict(factor):
    global FACTOR2INDEX_MAP, ZERO_FACTOR_INDEX
    if FACTOR2INDEX_MAP.get(factor) is None:
        FACTOR2INDEX_MAP[factor] = len(FACTOR2INDEX_MAP)
        if factor == 0:
            ZERO_FACTOR_INDEX = FACTOR2INDEX_MAP[factor]
def prime_init_factors(cell_weights, edge_weights):
    '''
    prime init factors for the cell weights and edge weights (including expression with symbols)
    all factors are stored in FACTOR_DICT
    '''
    for w in cell_weights:
        factored_list = factor_list(w)
        coef = factored_list[0]
        syms = factored_list[1]
        for k,_ in factorint(coef).items():
            update_factor_dict(k)
        for sym in syms:
            update_factor_dict(sym[0])
    for rs in edge_weights:
        for r in rs:
            factored_list = factor_list(r)
            coef = factored_list[0]
            syms = factored_list[1]
            for k,_ in factorint(coef).items():
                update_factor_dict(k)
            for sym in syms:
                update_factor_dict(sym[0])

def get_init_factor_set(cell_weights, edge_weights):
    cell_factor_set = []
    for w in cell_weights:
        vector = [0] * len(FACTOR2INDEX_MAP)
        factored_list = factor_list(w)
        coef = factored_list[0]
        syms = factored_list[1]
        for k,v in factorint(coef).items():
            vector[FACTOR2INDEX_MAP[k]] = v
        for sym in syms:
            vector[FACTOR2INDEX_MAP[sym[0]]] = int(sym[1])
        cell_factor_set.append(tuple(vector))
    global FACTOR_ADJ_MAT
    for i in range(len(edge_weights)):
        rs = edge_weights[i]
        vecs = []
        for j in range(len(rs)):
            r = rs[j]
            vector = [0] * len(FACTOR2INDEX_MAP)
            factored_list = factor_list(r)
            coef = factored_list[0]
            syms = factored_list[1]
            for k,v in factorint(coef).items():
                vector[FACTOR2INDEX_MAP[k]] = v
            for sym in syms:
                vector[FACTOR2INDEX_MAP[sym[0]]] = int(sym[1])
            vecs.append(tuple(vector))
        FACTOR_ADJ_MAT.append(vecs)
    return cell_factor_set

def get_vertex_color(weight):
    global VERTEX_COLOR_NO
    if weight not in VERTEX_WEIGHT2COLOR_MAP:
        VERTEX_WEIGHT2COLOR_MAP[weight] = VERTEX_COLOR_NO
        VERTEX_COLOR_NO += 1
    return VERTEX_WEIGHT2COLOR_MAP[weight]

def edgeWeight_To_edgeColor():
    global EDGE_COLOR_NUM, EDGE_WEIGHT2COLOR_MAP, COLOR_ADJ_MAT
    EDGE_COLOR_NUM = 0
    EDGE_WEIGHT2COLOR_MAP = {1:0}
    COLOR_ADJ_MAT = []
    
    for lst in ORI_WEIGHT_ADJ_MAT:
        tmp_list = []
        for w in lst:
            if w not in EDGE_WEIGHT2COLOR_MAP:
                EDGE_COLOR_NUM += 1
                EDGE_WEIGHT2COLOR_MAP[w] = EDGE_COLOR_NUM
            tmp_list.append(EDGE_WEIGHT2COLOR_MAP[w])
        COLOR_ADJ_MAT.append(tmp_list)
    
    global LAYERS_NUM_FOR_CONVERT, EXT_CELLS_NUM
    LAYERS_NUM_FOR_CONVERT = math.ceil(math.log2(EDGE_COLOR_NUM+1))
    EXT_CELLS_NUM = CELLS_NUM * LAYERS_NUM_FOR_CONVERT
    
def cellWeight_To_vertexColor(cell_weights):
    vertex_colors = []
    for w in cell_weights:
        vertex_colors.append(get_vertex_color(w))
    color_dict = Counter(vertex_colors)
    color_kind = tuple(sorted(color_dict))
    color_count = tuple(color_dict[num] for num in color_kind)
    return vertex_colors, color_kind, color_count

def calculate_adjacency_dict():
    # Generate new edges
    adjacency_dict = {}
    for i in range(EXT_CELLS_NUM):
        adjacency_dict[i] = []
    
    c2layers = {}
    for k in range(EDGE_COLOR_NUM+1):
        bi = bin(k)
        bi = bi[2:]
        bi = bi[::-1]
        layers = [i for i in range(len(bi)) if bi[i] == '1']
        c2layers[k] = layers
    
    for i in range(CELLS_NUM):
        for j in range(CELLS_NUM):
            layers = c2layers[COLOR_ADJ_MAT[i][j]]
            for l in layers:
                adjacency_dict[l*CELLS_NUM+i].append(l*CELLS_NUM+j)
    
    # The vertical threads (each corresponding to one vertex of the original graph) 
    # can be connected using either paths or cliques.
    for i in range(CELLS_NUM):
        clique = [i + j*CELLS_NUM for j in range(LAYERS_NUM_FOR_CONVERT)]
        for ii in clique:
            for jj in clique:
                if ii == jj:
                    continue
                adjacency_dict[ii].append(jj)
    global ADJACENCY_DICT
    ADJACENCY_DICT = adjacency_dict

def create_graph():
    global GRAPH
    GRAPH = pynauty.Graph(EXT_CELLS_NUM, 
                          directed=False, 
                          adjacency_dict=ADJACENCY_DICT)

def adjust_vertex_coloring(colored_vertices):
    '''
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
        print(new_colored_vertices)  # [0, 1, 0, 2, 1, 0]
        print(num_color)  # 3
    '''
    color_map = {}
    new_colored_vertices = []
    num_color = 0
    for c in colored_vertices:
        if c not in color_map:
            color_map[c] = num_color
            num_color += 1
        new_colored_vertices.append(color_map[c])
    return new_colored_vertices, num_color

def extend_vertex_coloring(colored_vertices, no_color):
    '''
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
    '''
    # Extend the vertex set to convert colored edge
    ext_colored_vertices = []
    for i in range(LAYERS_NUM_FOR_CONVERT):
        ext_colored_vertices += [x + no_color * i for x in colored_vertices]
    
    # Get color set of vertices
    no_color *= LAYERS_NUM_FOR_CONVERT
    vertex_coloring = [ set() for _ in range(no_color)]
    for i in range(len(ext_colored_vertices)):
        c = ext_colored_vertices[i]
        vertex_coloring[c].add(i)
    
    return vertex_coloring

def update_graph(colored_vertices):
    # for speed up, we have modified the function 'set_vertex_coloring' in graph.py of pynauty
    GRAPH.set_vertex_coloring(colored_vertices)
    return GRAPH

# not used
def get_gcd(cell_weights, cell_factor_tuple_list):
    gcd = Rational(1,1)
    gcd_vector = [0 for _ in range(len(FACTOR2INDEX_MAP))]
    for i in range(len(FACTOR2INDEX_MAP)):
        gcd_vector[i] = min([cell_factor_tuple_list[j][i] for j in range(CELLS_NUM)])
    if sum(gcd_vector) > 0:
        for i in range(len(cell_factor_tuple_list)):
            l_new = [x-y for x, y in zip(cell_factor_tuple_list[i], gcd_vector)]
            cell_factor_tuple_list[i] = tuple(l_new)
        for k,v in FACTOR2INDEX_MAP.items():
            if gcd_vector[v] > 0:
                gcd = expand(gcd * k**gcd_vector[v])
        for j in range(CELLS_NUM):
            cell_weights[j] = expand(cell_weights[j] / gcd)
    return gcd, cell_weights, cell_factor_tuple_list

# if all weights are integers, we can use this function to speed up by using cardinalities of fasctors
def dfs_wfomc(cell_weights, domain_size, cell_factor_tuple_list):  
    res = 0
    for l in range(CELLS_NUM):
        w_l = cell_weights[l]
        new_cell_weights = [cell_weights[i] * ORI_WEIGHT_ADJ_MAT[l][i] for i in range(CELLS_NUM)]
        if domain_size - 1 == 1:
            value = sum(new_cell_weights)   
        else:
            new_cell_factor_tuple_list = []
            for i in range(CELLS_NUM):
                new_cell_factor_tuple_list.append([x+y for x, y in zip(cell_factor_tuple_list[i], FACTOR_ADJ_MAT[l][i])])
                if ZERO_FACTOR_INDEX >= 0 and new_cell_factor_tuple_list[-1][ZERO_FACTOR_INDEX] > 0:
                    new_cell_factor_tuple_list[-1][ZERO_FACTOR_INDEX] = 1
                new_cell_factor_tuple_list[-1] = tuple(new_cell_factor_tuple_list[-1])
            # gcd, new_cell_weights, new_cell_factor_tuple_list = get_gcd(new_cell_weights, new_cell_factor_tuple_list)
            original_vertex_colors, vertex_color_kind, vertex_color_count = cellWeight_To_vertexColor(new_cell_factor_tuple_list) # convert cell weights to vertex colors
            if ENABLE_ISOMORPHISM:
                adjust_vertex_colors, no_color = adjust_vertex_coloring(original_vertex_colors) # adjust the color no. of vertices to make them start from 0 and be continuous
                if tuple(adjust_vertex_colors) not in CACHE_FOR_NAUTY:
                    can_label = pynauty.certificate(update_graph(extend_vertex_coloring(adjust_vertex_colors, no_color)))
                    CACHE_FOR_NAUTY[tuple(adjust_vertex_colors)] = can_label
                else:
                    can_label = CACHE_FOR_NAUTY[tuple(adjust_vertex_colors)]
            else:
                can_label = tuple(original_vertex_colors)
            
            value = IG_CACHE.get(domain_size-1, vertex_color_kind, vertex_color_count, can_label)
            if value is None:
                value = dfs_wfomc(new_cell_weights, domain_size - 1, new_cell_factor_tuple_list)
                value = expand(value)
                IG_CACHE.set(domain_size-1, vertex_color_kind, vertex_color_count, can_label, value)
        res += w_l * value # * expand(gcd**(domain_size - 1))
    return res

def dfs_wfomc_real(cell_weights, domain_size, node: TreeNode = None):
    res = 0
    for l in range(CELLS_NUM):
        w_l = cell_weights[l]
        new_cell_weights = [expand(cell_weights[i] * ORI_WEIGHT_ADJ_MAT[l][i]) for i in range(CELLS_NUM)]
        if PRINT_TREE:
            node.cell_to_children[l] = TreeNode(new_cell_weights, node.depth+1)
        if domain_size - 1 == 1:
            value = sum(new_cell_weights)   
        else:
            original_vertex_colors, vertex_color_kind, vertex_color_count = cellWeight_To_vertexColor(new_cell_weights) # convert cell weights to vertex colors
            if ENABLE_ISOMORPHISM:
                adjust_vertex_colors, no_color = adjust_vertex_coloring(original_vertex_colors) # # adjust the color no. of vertices to make them start from 0 and be continuous 
                if tuple(adjust_vertex_colors) not in CACHE_FOR_NAUTY:
                    can_label = pynauty.certificate(update_graph(extend_vertex_coloring(adjust_vertex_colors, no_color)))
                    CACHE_FOR_NAUTY[tuple(adjust_vertex_colors)] = can_label
                else:
                    can_label = CACHE_FOR_NAUTY[tuple(adjust_vertex_colors)]
            else:
                can_label = tuple(original_vertex_colors)
                
            value = IG_CACHE.get(domain_size-1, vertex_color_kind, vertex_color_count, can_label)
            if value is None:
                value = dfs_wfomc_real(new_cell_weights, domain_size - 1, node.cell_to_children[l] if PRINT_TREE else None)
                value = expand(value)
                IG_CACHE.set(domain_size-1, vertex_color_kind, vertex_color_count, can_label, value)
        res += w_l * value # * expand(gcd**(domain_size - 1))
    return res

def get_cache_size():
    total_size = 0
    for n_level in IG_CACHE.cache: # k0 is domain size
        for k1,v1 in n_level.items(): # k1 is color kind
            for k2,v2 in v1.items(): # k2 is color count
                for k3,v3 in v2.items(): # k3 is can_label
                    total_size += 1
    return total_size

def clean_global_variables():
    
    global PRINT_TREE, ROOT, IG_CACHE, ORI_WEIGHT_ADJ_MAT, \
        LAYERS_NUM_FOR_CONVERT, EDGE_COLOR_NUM, EDGE_WEIGHT2COLOR_MAP, \
            COLOR_ADJ_MAT, VERTEX_COLOR_NO, CELLS_NUM, EXT_CELLS_NUM, \
                VERTEX_WEIGHT2COLOR_MAP, ADJACENCY_DICT, CACHE_FOR_NAUTY, \
                    ENABLE_ISOMORPHISM, FACTOR2INDEX_MAP, ZERO_FACTOR_INDEX, FACTOR_ADJ_MAT
    
    PRINT_TREE = False
    ROOT = TreeNode([], 0)
    IG_CACHE = IsomorphicGraphCache()
    ORI_WEIGHT_ADJ_MAT = []
    LAYERS_NUM_FOR_CONVERT = 0
    EDGE_COLOR_NUM = 0
    EDGE_WEIGHT2COLOR_MAP = {1:0}
    COLOR_ADJ_MAT = []
    VERTEX_COLOR_NO = 0
    CELLS_NUM = 0
    EXT_CELLS_NUM = 0
    VERTEX_WEIGHT2COLOR_MAP = {}
    ADJACENCY_DICT = {}
    CACHE_FOR_NAUTY = {}
    ENABLE_ISOMORPHISM = True
    FACTOR2INDEX_MAP = {}
    ZERO_FACTOR_INDEX = -1
    FACTOR_ADJ_MAT = []

def recursive_wfomc(formula: QFFormula,
                  domain: set[Const],
                  get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                  leq_pred: Pred,
                  real_version: bool = True) -> RingElement:
    domain_size = len(domain)
    res = Rational(0, 1)
    for cell_graph, weight in build_cell_graphs(
        formula, get_weight, leq_pred=leq_pred
    ):
        cell_weights = cell_graph.get_all_weights()[0]
        edge_weights = cell_graph.get_all_weights()[1]
        
        clean_global_variables()
        
        IG_CACHE.init(domain_size)
        global ORI_WEIGHT_ADJ_MAT, CELLS_NUM
        # not change in the same problem
        CELLS_NUM = len(cell_weights)
        ORI_WEIGHT_ADJ_MAT = edge_weights
        edgeWeight_To_edgeColor()
        calculate_adjacency_dict()
        create_graph()
        
        # disable isomorphism
        # global ENABLE_ISOMORPHISM
        # ENABLE_ISOMORPHISM = False
        
        if not real_version:
            prime_init_factors(cell_weights, edge_weights)
            cell_factor_tuple_list = get_init_factor_set(cell_weights, edge_weights)
            res_ = dfs_wfomc(cell_weights, domain_size, cell_factor_tuple_list)
        else:
            global ROOT
            ROOT.cell_weights = cell_weights
            res_ = dfs_wfomc_real(cell_weights, domain_size, ROOT)
            if PRINT_TREE:
                print_tree(ROOT) 
        res = res + weight * res_
        print(weight * res_)
    return res