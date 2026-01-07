from collections import defaultdict
import pandas as pd
import functools
import networkx as nx
from itertools import product
from typing import Callable, Generator
from logzero import logger
from copy import deepcopy
from wfomc.fol import AtomicFormula, Const, Pred, QFFormula, a, b, c
from wfomc.network import PartitionConstraint
from wfomc.utils import Rational, RingElement, MultinomialCoefficients
from .components import Cell, TwoTable
from .utils import conditional_on

"""
cell_graph.py 是 WFOMC（加权一阶模型计数）算法的核心数据结构模块。它的主要作用是将一个抽象的一阶逻辑问题转换成一个具体的、可计算的图结构，即**“单元格图”（Cell Graph）**。

这个过程是“提升推理”（Lifted Inference）的关键步骤，其思想如下：

单元格（Cell）: 将论域中所有具有相同性质（满足相同的一元谓词）的元素归为一类，这一类就称为一个“单元格”或“1-类型”。
单元格图（Cell Graph）:
    图的节点是所有可能的单元格。
    每个节点有一个权重，代表属于该单元格的单个元素的权重。
    图的边代表两个单元格之间的交互关系。边的权重（在代码中由 TwoTable 表示）描述了从这两个单元格中各取一个元素时，它们之间满足二元谓词的加权模型数。
    通过构建这个图，算法就可以从对单个元素的推理提升到对“单元格类型”的推理，从而高效地处理大论域问题。

文件主要包含三个核心类和一个工厂函数：
    CellGraph: 基础的单元格图实现。
    OptimizedCellGraph: 针对 CellGraph 的优化版本，通过识别和利用对称性（cliques）和独立性来加速计算，用于 FastWFOMC 算法。
    OptimizedCellGraphWithPC: 在 OptimizedCellGraph 的基础上，增加了对划分约束（Partition Constraints）的支持。
    build_cell_graphs: 一个工厂函数，根据输入参数决定创建哪种类型的 CellGraph 实例。
"""
class CellGraph(object):
    """
    Cell graph that handles cells and the wmc between them.  处理单元格（cells）及其之间加权模型计数（WMC）的单元格图。
    """

    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                 leq_pred: Pred = None):
        """
        Cell graph that handles cells (1-types) and the WMC between them

        :param sentence QFFormula: the sentence in the form of quantifier-free formula
        :param get_weight Callable[[Pred], Tuple[RingElement, RingElement]]: the weighting function
        :param conditional_formulas list[CNF]: the optional conditional formula appended in WMC computing
        """
        self.formula: QFFormula = formula
        self.get_weight: Callable[[Pred],
                                  tuple[RingElement, RingElement]] = get_weight
        self.leq_pred: Pred = leq_pred
        self.preds: tuple[Pred] = tuple(self.formula.preds()) # 提取公式中所有的谓词，并存为一个元组。
        logger.debug('prednames: %s', self.preds)

        # --- 实例化公式 ---
        gnd_formula_ab1: QFFormula = self._ground_on_tuple(
            self.formula, a, b
        ) # 为了计算单元格（1-类型）和它们之间的关系（2-类型），需要将公式实例化。# 将公式中的变量替换为 (a, b)
        gnd_formula_ab2: QFFormula = self._ground_on_tuple(
            self.formula, b, a
        )# 将公式中的变量替换为 (b, a)，以保证对称性
        self.gnd_formula_ab: QFFormula = \
            gnd_formula_ab1 & gnd_formula_ab2 # `gnd_formula_ab` 用于计算任意两个元素之间的交互，是两者的合取。
        self.gnd_formula_cc: QFFormula = self._ground_on_tuple(
            self.formula, c
        )# `gnd_formula_cc` 用于定义单个元素的类型（单元格），将变量替换为 (c, c)。
        if self.leq_pred is not None: # 如果存在线性序，需要将序关系也加入到实例化公式中。
            self.gnd_formula_cc = self.gnd_formula_cc & self.leq_pred(c, c) # 单个元素必须满足 c <= c (自反性)。
            self.gnd_formula_ab = self.gnd_formula_ab & \
                self.leq_pred(b, a) & \
                (~self.leq_pred(a, b)) # 两个不同元素 a, b 必须满足 b <= a 且 a < b 不成立，这强制了一个顺序。
        logger.info('ground a b: %s', self.gnd_formula_ab)
        logger.info('ground c: %s', self.gnd_formula_cc)

        # build cells
        self.cells: list[Cell] = self._build_cells() # `_build_cells` 根据 `gnd_formula_cc` 找出所有可能的单元格类型。
        # filter cells
        logger.info('the number of valid cells: %s',
                    len(self.cells))

        logger.info('computing cell weights')
        self.cell_weights: dict[Cell, RingElement] = self._compute_cell_weights() # `_compute_cell_weights` 计算每个单元格的权重。
        logger.info('computing two table weights')
        self.two_tables: dict[tuple[Cell, Cell],
                              TwoTable] = self._build_two_tables() # `_build_two_tables` 计算任意两个单元格之间的交互权重。

    def _ground_on_tuple(self, formula: QFFormula,
                         c1: Const, c2: Const = None) -> QFFormula:
        """辅助函数，用常量 (c1, c2) 替换公式中的变量。"""
        variables = formula.vars() # 从输入的公式中提取出所有自由变量的集合。例如，对于公式 P(x) & R(x, y)，`variables` 将是 {'x', 'y'}。
        if len(variables) > 2:
            raise RuntimeError(
                "Can only ground out FO2"
            )
        if len(variables) == 1: # 情况1：公式只有一个变量，例如 P(x)。
            constants = [c1] # 那么这个变量将被替换为第一个常量 `c1`。
        else: # 情况2：公式有两个或零个变量。
            if c2 is not None: # 如果第二个常量 `c2` 被提供了（即不是 None）。
                constants = [c1, c2]
            else: # 如果第二个常量 `c2` 未提供。
                constants = [c1, c1]
        substitution = dict(zip(variables, constants)) # 创建一个替换字典。`zip` 会将变量列表和常量列表配对。例如，如果 variables 是 (x, y) 而 constants 是 [a, b]，字典将是 {'x': a, 'y': b}。
        gnd_formula = formula.substitute(substitution) # 调用公式对象的 `substitute` 方法，传入替换字典。这个方法会返回一个新的、变量已被替换为常量的公式对象。
        # # NOTE: workaround for the case where ground binary atoms not appearing in the formula
        # if c2 is not None:
        #     binary_preds = list(filter(
        #         lambda x: x.arity == 2, gnd_formula.preds()
        #     ))
        #     for pred in binary_preds:
        #         atom = pred(c1, c2)
        #         if atom not in gnd_formula.atoms():
        #             gnd_formula = gnd_formula & (atom | ~atom)
        #     # NOTE: end workaround
        return gnd_formula

    def show(self):
        logger.info(str(self))

    def __str__(self):
        s = 'CellGraph:\n'
        s += 'predicates: {}\n'.format(self.preds)
        cell_weight_df = []
        twotable_weight_df = []
        for _, cell1 in enumerate(self.cells):
            cell_weight_df.append(
                [str(cell1), self.get_cell_weight(cell1)]
            )
            twotable_weight = []
            for _, cell2 in enumerate(self.cells):
                # if idx1 < idx2:
                #     twotable_weight.append(0)
                #     continue
                twotable_weight.append(
                    self.get_two_table_weight(
                        (cell1, cell2))
                )
            twotable_weight_df.append(twotable_weight)
        cell_str = [str(cell) for cell in self.cells]
        cell_weight_df = pd.DataFrame(cell_weight_df, index=None,
                                      columns=['Cell', 'Weight'])
        twotable_weight_df = pd.DataFrame(twotable_weight_df, index=cell_str,
                                          columns=cell_str)
        s += 'cell weights: \n'
        s += cell_weight_df.to_markdown() + '\n'
        s += '2table weights: \n'
        s += twotable_weight_df.to_markdown()
        return s

    def __repr__(self):
        return str(self)

    def get_cells(self, cell_filter: Callable[[Cell], bool] = None) -> list[Cell]:
        """获取单元格列表，可选择性地进行过滤。"""
        if cell_filter is None:
            return self.cells
        return list(filter(cell_filter, self.cells))

    @functools.lru_cache(maxsize=None, typed=True)
    def get_cell_weight(self, cell: Cell) -> RingElement:
        """获取单个单元格的权重（带缓存）。"""
        if cell not in self.cell_weights:
            logger.warning(
                "Cell %s not found", cell
            )
            return 0
        return self.cell_weights.get(cell)

    def _check_existence(self, cells: tuple[Cell, Cell]):
        """检查给定的单元格对是否存在于 two_tables 中。"""
        if cells not in self.two_tables:
            raise ValueError(
                f"Cells {cells} not found, note that the order of cells matters!"
            )

    @functools.lru_cache(maxsize=None, typed=True)
    def get_two_table_weight(self, cells: tuple[Cell, Cell],
                             evidences: frozenset[AtomicFormula] = None) -> RingElement:
        """获取两个单元格之间的交互权重（带缓存）。"""
        self._check_existence(cells)
        return self.two_tables.get(cells).get_weight(evidences)

    def get_all_weights(self) -> tuple[list[RingElement], list[RingElement]]:
        cell_weights = []
        twotable_weights = []
        for cell_i in self.cells:
            w = self.get_cell_weight(cell_i)
            cell_weights.append(w)
            twotable_weight = []
            for cell_j in self.cells:
                r = self.get_two_table_weight((cell_i, cell_j))
                twotable_weight.append(r)
            twotable_weights.append(twotable_weight)
        return cell_weights, twotable_weights

    @functools.lru_cache(maxsize=None, typed=True)
    def satisfiable(self, cells: tuple[Cell, Cell],
                    evidences: frozenset[AtomicFormula] = None) -> bool:
        self._check_existence(cells)
        return self.two_tables.get(cells).satisfiable(evidences)

    @functools.lru_cache(maxsize=None)
    def get_two_tables(self, cells: tuple[Cell, Cell],
                       evidences: frozenset[AtomicFormula] = None) \
            -> tuple[frozenset[AtomicFormula], RingElement]:
        self._check_existence(cells)
        return self.two_tables.get(cells).get_two_tables(evidences)

    def _build_cells(self):
        """构建所有可能的单元格（1-类型）。"""
        cells = []
        code = {}
        for model in self.gnd_formula_cc.models(): # 遍历单元素接地公式 `gnd_formula_cc` 的所有模型。每个模型都代表了一种合法的元素类型。
            for lit in model: # 将模型（一组真值赋值）转换成一个编码（code）。
                code[lit.pred] = lit.positive
            cells.append(Cell(tuple(code[p] for p in self.preds), self.preds)) # 用这个编码创建一个 Cell 对象。
        return cells

    def _compute_cell_weights(self):
        weights = dict()
        for cell in self.cells:
            weight = Rational(1, 1)
            for i, pred in zip(cell.code, cell.preds):
                assert pred.arity > 0, "Nullary predicates should have been removed"
                if i:
                    weight = weight * self.get_weight(pred)[0]
                else:
                    weight = weight * self.get_weight(pred)[1]
            weights[cell] = weight
        return weights

    @functools.lru_cache(maxsize=None)
    def get_nullary_weight(self, cell: Cell) -> RingElement:
        weight = Rational(1, 1)
        for i, pred in zip(cell.code, cell.preds):
            if pred.arity == 0:
                if i:
                    weight = weight * self.get_weight(pred)[0]
                else:
                    weight = weight * self.get_weight(pred)[1]
        return weight

    def _build_two_tables(self):
        # build a pd.DataFrame containing all model as well as the weight
        models = dict()
        gnd_lits = self.gnd_formula_ab.atoms()
        gnd_lits = gnd_lits.union(
            frozenset(map(lambda x: ~x, gnd_lits))
        )
        for model in self.gnd_formula_ab.models():
            weight = Rational(1, 1)
            for lit in model:
                # ignore the weight appearing in cell weight
                if (not (len(lit.args) == 1 or all(arg == lit.args[0]
                                                   for arg in lit.args))):
                    weight *= (self.get_weight(lit.pred)[0] if lit.positive else
                               self.get_weight(lit.pred)[1])
            models[frozenset(model)] = weight
        # build twotable tables
        tables = dict()
        for i, cell in enumerate(self.cells):
            models_1 = conditional_on(models, gnd_lits, cell.get_evidences(a))
            for j, other_cell in enumerate(self.cells):
                # NOTE: leq is sensitive to the order of cells
                if i > j and self.leq_pred is None: # 当 leq_pred 未定义时，(cell, other_cell) 和 (other_cell, cell) 之间的关系被认为是还未定义的。因为当 i > j 时，索引 j 对应的 other_cell 和索引 i 对应的 cell 组成的元组 (other_cell, cell) 已经在之前的循环中计算过了。
                    tables[(cell, other_cell)] = tables[(other_cell, cell)] # B(X, Y) 的约束是针对“出度”的。它要求每个节点 X 的“出度”（B(X, Y) 为真的 Y 的数量）必须是奇数。但是，它对节点的“入度”（B(Y, X) 为真的 Y 的数量）没有任何要求。因此，B(X, Y) 和 B(Y, X) 的真值可能是不同的。
                models_2 = conditional_on(models_1, gnd_lits,
                                          other_cell.get_evidences(b))
                tables[(cell, other_cell)] = TwoTable(
                    models_2, gnd_lits
                )
        return tables


class OptimizedCellGraph(CellGraph):
    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                 domain_size: int,
                 modified_cell_symmetry: bool = False):
        """
        Optimized cell graph for FastWFOMC
        :param formula: the formula to be grounded
        :param get_weight: a function that returns the weight of a predicate
        :param domain_size: the domain size
        """
        super().__init__(formula, get_weight)
        self.modified_cell_symmetry = modified_cell_symmetry
        self.domain_size: int = domain_size
        MultinomialCoefficients.setup(self.domain_size)

        if self.modified_cell_symmetry:
            i1_ind_set, i2_ind_set, nonind_set = self.find_independent_sets()
            self.cliques, [self.i1_ind, self.i2_ind, self.nonind] = \
                self.build_symmetric_cliques_in_ind([i1_ind_set, i2_ind_set, nonind_set])
            self.nonind_map: dict[int, int] = dict(zip(self.nonind, range(len(self.nonind))))
        else:
            self.cliques: list[list[Cell]] = self.build_symmetric_cliques()
            self.i1_ind, self.i2_ind, self.ind, self.nonind \
                = self.find_independent_cliques()
            self.nonind_map: dict[int, int] = dict(
                zip(self.nonind, range(len(self.nonind))))

        logger.info("Found i1 independent cliques: %s", self.i1_ind)
        logger.info("Found i2 independent cliques: %s", self.i2_ind)
        logger.info("Found non-independent cliques: %s", self.nonind)

        self.term_cache = dict()

    def build_symmetric_cliques(self) -> list[list[Cell]]:
        cliques: list[list[Cell]] = []
        cells = deepcopy(self.get_cells())
        while len(cells) > 0:
            cell = cells.pop()
            clique = [cell]
            for other_cell in cells:
                if self._matches(clique, other_cell):
                    clique.append(other_cell)
            for other_cell in clique[1:]:
                cells.remove(other_cell)
            cliques.append(clique)
        cliques.sort(key=len)
        logger.info("Built %s symmetric cliques: %s", len(cliques), cliques)
        return cliques

    def build_symmetric_cliques_in_ind(self, cell_indices_list) -> \
            tuple[list[list[Cell]], list[list[int]]]:
        i1_ind_set = deepcopy(cell_indices_list[0])
        cliques: list[list[Cell]] = []
        ind_idx: list[list[int]] = []
        for cell_indices in cell_indices_list:
            idx_list = []
            while len(cell_indices) > 0:
                cell_idx = cell_indices.pop()
                clique = [self.cells[cell_idx]]
                # for cell in I1 independent set, we dont need to built sysmmetric cliques
                if cell_idx not in i1_ind_set:
                    for other_cell_idx in cell_indices:
                        other_cell = self.cells[other_cell_idx]
                        if self._matches(clique, other_cell):
                            clique.append(other_cell)
                    for other_cell in clique[1:]:
                        cell_indices.remove(self.cells.index(other_cell))
                cliques.append(clique)
                idx_list.append(len(cliques) - 1)
            ind_idx.append(idx_list)
        logger.info("Built %s symmetric cliques: %s", len(cliques), cliques)
        return cliques, ind_idx

    def find_independent_sets(self) -> tuple[list[int], list[int], list[int], list[int]]:
        g = nx.Graph()
        g.add_nodes_from(range(len(self.cells)))
        for i in range(len(self.cells)):
            for j in range(i + 1, len(self.cells)):
                if self.get_two_table_weight(
                        (self.cells[i], self.cells[j])
                ) != Rational(1, 1):
                    g.add_edge(i, j)

        self_loop = set()
        for i in range(len(self.cells)):
            if self.get_two_table_weight((self.cells[i], self.cells[i])) != Rational(1, 1):
                self_loop.add(i)

        non_self_loop = g.nodes - self_loop
        if len(non_self_loop) == 0:
            i1_ind = set()
        else:
            i1_ind = set(nx.maximal_independent_set(g.subgraph(non_self_loop)))
        g_ind = set(nx.maximal_independent_set(g, nodes=i1_ind))
        i2_ind = g_ind.difference(i1_ind)
        non_ind = g.nodes - i1_ind - i2_ind
        logger.info("Found i1 independent set: %s", i1_ind)
        logger.info("Found i2 independent set: %s", i2_ind)
        logger.info("Found non-independent set: %s", non_ind)
        return list(i1_ind), list(i2_ind), list(non_ind)

    def find_independent_cliques(self) -> tuple[list[int], list[int], list[int], list[int]]:
        g = nx.Graph()
        g.add_nodes_from(range(len(self.cliques)))
        for i in range(len(self.cliques)):
            for j in range(i + 1, len(self.cliques)):
                if self.get_two_table_weight(
                        (self.cliques[i][0], self.cliques[j][0])
                ) != Rational(1, 1):
                    g.add_edge(i, j)

        self_loop = set()
        for i in range(len(self.cliques)):
            for j in range(1, self.domain_size + 1):
                if self.get_J_term(i, j) != Rational(1, 1):
                    self_loop.add(i)
                    break

        non_self_loop = g.nodes - self_loop
        if len(non_self_loop) == 0:
            g_ind = set()
        else:
            g_ind = set(nx.maximal_independent_set(g.subgraph(non_self_loop)))
        i2_ind = g_ind.intersection(self_loop)
        i1_ind = g_ind.difference(i2_ind)
        non_ind = g.nodes - i1_ind - i2_ind
        return list(i1_ind), list(i2_ind), list(g_ind), list(non_ind)

    def _matches(self, clique, other_cell) -> bool:
        cell = clique[0]
        if not self.modified_cell_symmetry:
            if self.get_cell_weight(cell) != self.get_cell_weight(other_cell) or \
                    self.get_two_table_weight((cell, cell)) != self.get_two_table_weight((other_cell, other_cell)):
                return False

        if len(clique) > 1:
            third_cell = clique[1]
            r = self.get_two_table_weight((cell, third_cell))
            for third_cell in clique:
                if r != self.get_two_table_weight((other_cell, third_cell)):
                    return False

        for third_cell in self.get_cells():
            if other_cell == third_cell or third_cell in clique:
                continue
            r = self.get_two_table_weight((cell, third_cell))
            if r != self.get_two_table_weight((other_cell, third_cell)):
                return False
        return True

    def setup_term_cache(self):
        self.term_cache = dict()

    def get_term(self, iv: int, bign: int, partition: tuple[int]) -> RingElement:
        if (iv, bign) in self.term_cache:
            return self.term_cache[(iv, bign)]

        if iv == 0:
            accum = Rational(0, 1)
            for j in self.i1_ind:
                tmp = self.get_cell_weight(self.cliques[j][0])
                for i in self.nonind:
                    tmp = tmp * self.get_two_table_weight(
                        (self.cliques[i][0], self.cliques[j][0])) ** partition[self.nonind_map[i]]
                accum = accum + tmp
            accum = accum ** (self.domain_size - sum(partition) - bign)
            self.term_cache[(iv, bign)] = accum
            return accum
        else:
            sumtoadd = 0
            s = self.i2_ind[len(self.i2_ind) - iv]
            for nval in range(self.domain_size - sum(partition) - bign + 1):
                smul = MultinomialCoefficients.comb(
                    self.domain_size - sum(partition) - bign, nval
                )
                smul = smul * self.get_J_term(s, nval)
                if not self.modified_cell_symmetry:
                    smul = smul * self.get_cell_weight(self.cliques[s][0]) ** nval

                for i in self.nonind:
                    smul = smul * self.get_two_table_weight(
                        (self.cliques[i][0], self.cliques[s][0])
                    ) ** (partition[self.nonind_map[i]] * nval)
                smul = smul * self.get_term(
                    iv - 1, bign + nval, partition
                )
                sumtoadd = sumtoadd + smul
            self.term_cache[(iv, bign)] = sumtoadd
            return sumtoadd

    @functools.lru_cache(maxsize=None)
    def get_J_term(self, l: int, nhat: int) -> RingElement:
        if len(self.cliques[l]) == 1:
            thesum = self.get_two_table_weight(
                (self.cliques[l][0], self.cliques[l][0])
            ) ** (int(nhat * (nhat - 1) / 2))
            if self.modified_cell_symmetry:
                thesum = thesum * self.get_cell_weight(self.cliques[l][0]) ** nhat
        else:
            thesum = self.get_d_term(l, nhat)
        return thesum

    @functools.lru_cache(maxsize=None)
    def get_d_term(self, l: int, n: int, cur: int = 0) -> RingElement:
        clique_size = len(self.cliques[l])
        r = self.get_two_table_weight((self.cliques[l][0], self.cliques[l][1]))
        s = self.get_two_table_weight((self.cliques[l][0], self.cliques[l][0]))
        if cur == clique_size - 1:
            if self.modified_cell_symmetry:
                w = self.get_cell_weight(self.cliques[l][cur]) ** n
                s = self.get_two_table_weight((self.cliques[l][cur], self.cliques[l][cur]))
                ret = w * s ** MultinomialCoefficients.comb(n, 2)
            else:
                ret = s ** MultinomialCoefficients.comb(n, 2)
        else:
            ret = 0
            for ni in range(n + 1):
                mult = MultinomialCoefficients.comb(n, ni)
                if self.modified_cell_symmetry:
                    w = self.get_cell_weight(self.cliques[l][cur]) ** ni
                    s = self.get_two_table_weight((self.cliques[l][cur], self.cliques[l][cur]))
                    mult = mult * w
                mult = mult * (s ** MultinomialCoefficients.comb(ni, 2))
                mult = mult * r ** (ni * (n - ni))
                mult = mult * self.get_d_term(l, n - ni, cur + 1)
                ret = ret + mult
        return ret


class OptimizedCellGraphWithPC(CellGraph):
    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                 domain_size: int,
                 partition_constraint: PartitionConstraint):
        """
        Optimized cell graph for FastWFOMC
        :param formula: the formula to be grounded
        :param get_weight: a function that returns the weight of a predicate
        :param domain_size: the domain size
        """
        super().__init__(formula, get_weight)
        self.domain_size: int = domain_size
        self.partition_constraint = partition_constraint

        i1_ind_set, i2_ind_set, nonind_set = self.find_independent_sets()
        # TODO incorporate partition constraint into i2_ind_set
        # for now, merge i2_ind_set into nonind_set
        self.i1_ind = i1_ind_set
        nonind_set = i2_ind_set + nonind_set
        self.cliques, self.nonind = \
            self.build_symmetric_cliques(nonind_set)
        self.nonind_map: dict[int, int] = dict(zip(self.nonind, range(len(self.nonind))))

        logger.info(f"Found i1 independent cliques: {self.i1_ind}")
        logger.info(f"Found non-independent cliques: {self.nonind}")
        self.term_cache = dict()

        # partition each clique according to the partition constraint
        # NOTE: ignore the empty set of cells
        # {clique_idx: [[cell_idx_in_clique for pred1],
        #               [cell_idx_in_clique for pred2],
        #                             ..
        #               [cell_idx_in_clique for predk]]}
        # partition constrained cells into cliques
        # {pred_idx: [clique_indices that pred1 appears,
        #             clique_indices that pred2 appears
        #                     ndices...
        #             clique_indices that pred3 appears]
        self.clique_partitions: dict[int, list[list[int]]] = dict()
        self.partition_cliques: dict[int, list[int]] = defaultdict(list)
        for idx, clique in enumerate(self.cliques):
            clique_partiton = list()
            for pred_idx, (pred, _) in enumerate(
                    self.partition_constraint.partition
            ):
                cell_indices = list()
                for cell_idx, cell in enumerate(clique):
                    if cell.is_positive(pred):
                        cell_indices.append(cell_idx)
                if len(cell_indices) > 0:
                    clique_partiton.append(cell_indices)
                    self.partition_cliques[pred_idx].append(idx)
            self.clique_partitions[idx] = clique_partiton
        logger.info("Clique partitions: %s", self.clique_partitions)
        logger.info("Partition cliques: %s", self.partition_cliques)

        self.i1_partition: list[list[int]] = list()
        for pred, _ in self.partition_constraint.partition:
            self.i1_partition.append(list(filter(
                lambda idx: self.cells[idx].is_positive(pred), self.i1_ind
            )))

    def set_clique_configs(self, clique_configs: dict[int, list[int]]):
        self.clique_configs = clique_configs
        self.ovarall_clique_config = list(sum(v) for _, v in self.clique_configs.items())
        logger.info('Clique configs: %s', self.clique_configs)
        logger.info('Overall clique config: %s', self.ovarall_clique_config)

    def find_independent_sets(self) -> tuple[list[int], list[int], list[int], list[int]]:
        # find the cell with partition constraint
        in_partition_cells = set()
        self.partition_constraint.partition = \
            sorted(self.partition_constraint.partition, key=lambda x: x[1], reverse=True)
        maximal_partition_pred = self.partition_constraint.partition[0][0]
        g = nx.Graph()
        g.add_nodes_from(range(len(self.cells)))
        for i in range(len(self.cells)):
            if self.cells[i].is_positive(maximal_partition_pred):
                in_partition_cells.add(i)
            for j in range(i + 1, len(self.cells)):
                if self.get_two_table_weight(
                        (self.cells[i], self.cells[j])
                ) != Rational(1, 1):
                    g.add_edge(i, j)

        self_loop = set()
        for i in range(len(self.cells)):
            if self.get_two_table_weight((self.cells[i], self.cells[i])) != Rational(1, 1):
                self_loop.add(i)

        non_self_loop = g.nodes - self_loop
        # NOTE: only consider the cells with partition constraint
        # ind_seed = non_self_loop & in_partition_cells
        if len(non_self_loop) == 0:
            i1_ind = set()
        else:
            if len(non_self_loop & in_partition_cells) != 0:
                in_partition_ind = set(nx.maximal_independent_set(g.subgraph(non_self_loop & in_partition_cells)))
                i1_ind = set(nx.maximal_independent_set(g.subgraph(non_self_loop), nodes=in_partition_ind))
            else:
                i1_ind = set(nx.maximal_independent_set(g.subgraph(non_self_loop)))
        g_ind = set(nx.maximal_independent_set(g, nodes=i1_ind))
        i2_ind = g_ind.difference(i1_ind)
        non_ind = g.nodes - i1_ind - i2_ind
        logger.info("Found i1 independent set: %s", i1_ind)
        logger.info("Found i2 independent set: %s", i2_ind)
        logger.info("Found non-independent set: %s", non_ind)
        return list(i1_ind), list(i2_ind), list(non_ind)

    def _matches(self, clique, other_cell) -> bool:
        cell = clique[0]
        if len(clique) > 1:
            third_cell = clique[1]
            r = self.get_two_table_weight((cell, third_cell))
            for third_cell in clique:
                if r != self.get_two_table_weight((other_cell, third_cell)):
                    return False

        for third_cell in self.get_cells():
            if other_cell == third_cell or third_cell in clique:
                continue
            r = self.get_two_table_weight((cell, third_cell))
            if r != self.get_two_table_weight((other_cell, third_cell)):
                return False
        return True

    def build_symmetric_cliques(self, cell_indices) -> \
            tuple[list[list[Cell]], list[int]]:
        cliques: list[list[Cell]] = []
        idx_list = []
        while len(cell_indices) > 0:
            cell_idx = cell_indices.pop()
            clique = [self.cells[cell_idx]]
            # for cell in I1 independent set, we dont need to built symmetric cliques
            for other_cell_idx in cell_indices:
                other_cell = self.cells[other_cell_idx]
                if self._matches(clique, other_cell):
                    clique.append(other_cell)
            for other_cell in clique[1:]:
                cell_indices.remove(self.cells.index(other_cell))
            cliques.append(clique)
            idx_list.append(len(cliques) - 1)
        logger.info("Built %s symmetric cliques: %s", len(cliques), cliques)
        return cliques, idx_list

    def get_i1_weight(self, i1_config: tuple[int],
                      config: tuple[int]) -> RingElement:
        ret = Rational(1, 1)
        for i1_inds, num in zip(self.i1_partition, i1_config):
            # NOTE: it means the config is not valid
            if len(i1_inds) == 0 and num > 0:
                return 0
            accum = Rational(0, 1)
            for i in i1_inds:
                tmp = self.get_cell_weight(self.cells[i])
                for j in self.nonind:
                    tmp = tmp * self.get_two_table_weight(
                        (self.cliques[j][0], self.cells[i])
                    ) ** config[self.nonind_map[j]]
                accum = accum + tmp
            ret = ret * (accum ** num)
        return ret

    @functools.lru_cache(maxsize=None)
    def get_J_term(self, l: int, clique_config: tuple[int]) -> RingElement:
        """
        clique_config: the partition config in the clique l
        """
        ret = Rational(1, 1)
        clique = self.cliques[l]
        clique_partition = self.clique_partitions[l]
        # the clique belongs to one pred
        if len(clique_partition) == 1:
            ret = self.get_partitioned_J_term(
                l, 0, clique_config[0]
            )
        else:
            # at least two cells in the clique
            r = self.get_two_table_weight((clique[0], clique[1]))
            sumn = 0
            for i, n1 in enumerate(clique_config):
                for j, n2 in enumerate(clique_config):
                    if i < j:
                        sumn = sumn + n1 * n2
            ret = ret * (r ** sumn)
            for par_idx in range(len(clique_partition)):
                ret = ret * self.get_partitioned_J_term(
                    l, par_idx, clique_config[par_idx]
                )
        return ret

    @functools.lru_cache(maxsize=None)
    def get_partitioned_J_term(self, l: int, par_idx: int, nhat: int):
        cell_indices_in_clique = self.clique_partitions[l][par_idx]
        clique = self.cliques[l]
        if len(cell_indices_in_clique) == 1:
            thesum = self.get_two_table_weight(
                (clique[cell_indices_in_clique[0]],
                 clique[cell_indices_in_clique[0]])
            ) ** MultinomialCoefficients.comb(nhat, 2)
            thesum = thesum * self.get_cell_weight(
                clique[cell_indices_in_clique[0]]
            ) ** nhat
        else:
            thesum = self.get_d_term(l, nhat, par_idx)
        return thesum

    @functools.lru_cache(maxsize=None)
    def get_d_term(self, l: int, n: int, par_idx: int, cur: int = 0) -> RingElement:
        cell_indices_in_clique = self.clique_partitions[l][par_idx]
        cell_index = cell_indices_in_clique[cur]
        cells_num = len(cell_indices_in_clique)
        clique = self.cliques[l]
        r = self.get_two_table_weight((clique[0], clique[1]))
        if cur == cells_num - 1:
            w = self.get_cell_weight(clique[cell_index]) ** n
            s = self.get_two_table_weight((clique[cell_index], clique[cell_index]))
            ret = w * s ** MultinomialCoefficients.comb(n, 2)
        else:
            ret = 0
            for ni in range(n + 1):
                mult = MultinomialCoefficients.comb(n, ni)
                w = self.get_cell_weight(clique[cell_index]) ** ni
                s = self.get_two_table_weight((clique[cell_index], clique[cell_index]))
                mult = mult * w
                mult = mult * (s ** MultinomialCoefficients.comb(ni, 2))
                mult = mult * r ** (ni * (n - ni))
                mult = mult * self.get_d_term(l, n - ni, par_idx, cur + 1)
                ret = ret + mult
        return ret


def build_cell_graphs(formula: QFFormula,
                      get_weight: Callable[[Pred],
                                           tuple[RingElement, RingElement]],
                      leq_pred: Pred = None,
                      optimized: bool = False,
                      domain_size: int = 0,
                      modified_cell_symmetry: bool = False,
                      partition_constraint: PartitionConstraint = None) \
        -> Generator[tuple[CellGraph, RingElement], None, None]:
            
    """
    这个函数是创建 CellGraph 对象的统一入口，它像一个工厂，根据传入的参数和公式的特性，决定“生产”哪一种 CellGraph（基础版、优化版、或带划分约束的优化版）。

    它的一个关键特性是处理0元谓词（Nullary Predicates），即没有参数的谓词，如 IsRaining。这些谓词本质上是命题变量，它们的真假与论域中的元素无关。如果公式中存在0元谓词，为了计算总的加权模型数，必须分别计算该谓词为真和为假两种情况下的模型数，然后加权求和。

    因此，这个函数被设计成一个生成器（Generator）。

    如果没有0元谓词，它只会 yield 一个 CellGraph 实例。
    如果有0元谓词，它会遍历0元谓词所有可能的真值组合，并为每一种组合 yield 一个对应的 CellGraph 实例及其权重。调用者需要将所有 yield 出来的结果进行加权求和，才能得到最终的总数。
    """
    # --- 1. 检查0元谓词 ---
    nullary_atoms = [atom for atom in formula.atoms() if atom.pred.arity == 0] # 从公式中找出所有0元谓词（arity == 0）。
    # --- 2. 情况一：没有0元谓词 ---
    if len(nullary_atoms) == 0: # 这是最简单的情况，只需要构建一个 CellGraph。
        logger.info('No nullary atoms found, building a single cell graph')
        if not optimized: # 根据 `optimized` 参数决定是创建基础版还是优化版。
            yield CellGraph(
                formula, get_weight, leq_pred
            ), Rational(1, 1) # 创建并 yield 一个基础的 CellGraph。权重为1，因为没有0元谓词的贡献。
        else: # 创建优化版 CellGraph。
            if partition_constraint is None: # 进一步根据 `partition_constraint` 是否存在，决定是创建  `OptimizedCellGraph` 还是 `OptimizedCellGraphWithPC`。
                yield OptimizedCellGraph(
                    formula, get_weight, domain_size, modified_cell_symmetry
                ), Rational(1, 1)
            else:
                yield OptimizedCellGraphWithPC(
                    formula, get_weight, domain_size, partition_constraint
                ), Rational(1, 1)
    else: # --- 3. 情况二：存在0元谓词 --- # 需要为0元谓词的每一种真值组合构建一个 CellGraph。
        logger.info('Found nullary atoms %s', nullary_atoms)
        for values in product(*([[True, False]] * len(nullary_atoms))): # `product(*([[True, False]] * len(nullary_atoms)))` 生成所有可能的真值组合。例如，如果有2个0元谓词，它会生成 (True, True), (True, False), (False, True), (False, False)。
            substitution = dict(zip(nullary_atoms, values))
            logger.info('Building cell graph with values %s', substitution)
            subs_formula = formula.sub_nullary_atoms(substitution).simplify()
            if not subs_formula.satisfiable():
                logger.info('Formula is unsatisfiable, skipping')
                continue
            if not optimized:
                cell_graph = CellGraph(
                    subs_formula, get_weight, leq_pred
                )
            else:
                if partition_constraint is None:
                    cell_graph = OptimizedCellGraph(
                        subs_formula, get_weight, domain_size, modified_cell_symmetry
                    )
                else:
                    cell_graph = OptimizedCellGraphWithPC(
                        subs_formula, get_weight, domain_size, partition_constraint
                    )
            weight = Rational(1, 1)
            for atom, val in zip(nullary_atoms, values):
                weight = weight * (get_weight(atom.pred)[0] if val else get_weight(atom.pred)[1])
            yield cell_graph, weight
