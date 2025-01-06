from __future__ import annotations
from collections import defaultdict

import pandas as pd
import functools
import networkx as nx
from itertools import product

from typing import Callable, Dict, FrozenSet, Generator, List, Tuple
from logzero import logger
from sympy import Poly
from copy import deepcopy
from wfomc.cell_graph.utils import conditional_on

from wfomc.fol.syntax import AtomicFormula, Const, Pred, QFFormula, a, b, c
from wfomc.network.constraint import PartitionConstraint
from wfomc.utils import Rational, RingElement
from wfomc.utils.multinomial import MultinomialCoefficients

from .components import Cell, TwoTable


class CellGraph(object):
    """
    Cell graph that handles cells and the wmc between them.
    """

    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], Tuple[RingElement, RingElement]],
                 leq_pred: Pred = None,
                 predecessor_pred: Pred = None):
        """
        Cell graph that handles cells (1-types) and the WMC between them

        :param sentence QFFormula: the sentence in the form of quantifier-free formula
        :param get_weight Callable[[Pred], Tuple[RingElement, RingElement]]: the weighting function
        :param conditional_formulas List[CNF]: the optional conditional formula appended in WMC computing
        """
        self.formula: QFFormula = formula
        self.get_weight: Callable[[Pred],
                                  Tuple[RingElement, RingElement]] = get_weight
        self.leq_pred: Pred = leq_pred
        self.predecessor_pred: Pred = predecessor_pred
        self.preds: Tuple[Pred] = tuple(self.formula.preds())
        logger.debug('prednames: %s', self.preds)

        gnd_formula_ab1: QFFormula = self._ground_on_tuple(
            self.formula, a, b
        )
        gnd_formula_ab2: QFFormula = self._ground_on_tuple(
            self.formula, b, a
        )
        self.gnd_formula_ab: QFFormula = \
            gnd_formula_ab1 & gnd_formula_ab2
        self.gnd_formula_cc: QFFormula = self._ground_on_tuple(
            self.formula, c
        )
        if self.leq_pred is not None:
            self.gnd_formula_cc = self.gnd_formula_cc & self.leq_pred(c, c)
            self.gnd_formula_ab = self.gnd_formula_ab & \
                self.leq_pred(b, a) & \
                (~self.leq_pred(a, b))
            if self.predecessor_pred is not None:
                self.gnd_formula_cc = self.gnd_formula_cc & (~self.predecessor_pred(c, c))
                self.gnd_formula_ab_with_pred = self.gnd_formula_ab & \
                    self.predecessor_pred(b, a) & \
                    (~self.predecessor_pred(a, b))
                self.gnd_formula_ab = self.gnd_formula_ab & \
                    (~self.predecessor_pred(b, a)) & \
                    (~self.predecessor_pred(a, b))
        logger.info('ground a b: %s', self.gnd_formula_ab)
        logger.info('ground c: %s', self.gnd_formula_cc)
        if self.predecessor_pred is not None:
            logger.info(f'ground a, b with predecessor: {self.gnd_formula_ab_with_pred}')

        # build cells
        self.cells: List[Cell] = self._build_cells()
        # filter cells
        logger.info('the number of valid cells: %s',
                    len(self.cells))

        logger.info('computing cell weights')
        self.cell_weights: Dict[Cell, Poly] = self._compute_cell_weights()
        logger.info('computing two table weights')
        self.two_tables: Dict[Tuple[Cell, Cell],
                              TwoTable] = self._build_two_tables(self.gnd_formula_ab)
        if self.predecessor_pred is not None:
            self.two_tables_with_pred: Dict[Tuple[Cell, Cell],
                                            TwoTable] = self._build_two_tables(
                self.gnd_formula_ab_with_pred
            )

    def _ground_on_tuple(self, formula: QFFormula,
                         c1: Const, c2: Const = None) -> QFFormula:
        variables = formula.vars()
        if len(variables) > 2:
            raise RuntimeError(
                "Can only ground out FO2"
            )
        if len(variables) == 1:
            constants = [c1]
        else:
            if c2 is not None:
                constants = [c1, c2]
            else:
                constants = [c1, c1]
        substitution = dict(zip(variables, constants))
        gnd_formula = formula.substitute(substitution)
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

    def get_cells(self, cell_filter: Callable[[Cell], bool] = None) -> List[Cell]:
        if cell_filter is None:
            return self.cells
        return list(filter(cell_filter, self.cells))

    @functools.lru_cache(maxsize=None, typed=True)
    def get_cell_weight(self, cell: Cell) -> Poly:
        if cell not in self.cell_weights:
            logger.warning(
                "Cell %s not found", cell
            )
            return 0
        return self.cell_weights.get(cell)

    def _check_existence(self, cells: Tuple[Cell, Cell]):
        if cells not in self.two_tables:
            raise ValueError(
                f"Cells {cells} not found, note that the order of cells matters!"
            )

    @functools.lru_cache(maxsize=None, typed=True)
    def get_two_table_weight(self, cells: Tuple[Cell, Cell],
                             evidences: FrozenSet[AtomicFormula] = None) -> RingElement:
        self._check_existence(cells)
        return self.two_tables.get(cells).get_weight(evidences)

    @functools.lru_cache(maxsize=None, typed=True)
    def get_two_table_with_pred_weight(self, cells: Tuple[Cell, Cell],
                             evidences: FrozenSet[AtomicFormula] = None) -> RingElement:
        self._check_existence(cells)
        return self.two_tables_with_pred.get(cells).get_weight(evidences)

    def get_all_weights(self) -> Tuple[List[RingElement], List[RingElement]]:
        cell_weights = []
        twotable_weights = []
        for cell_i in self.cells:
            cell_weights.append(self.get_cell_weight(cell_i))
            twotable_weight = []
            for cell_j in self.cells:
                twotable_weight.append(self.get_two_table_weight(
                    (cell_i, cell_j)
                ))
            twotable_weights.append(twotable_weight)
        return cell_weights, twotable_weights

    @functools.lru_cache(maxsize=None, typed=True)
    def satisfiable(self, cells: Tuple[Cell, Cell],
                    evidences: FrozenSet[AtomicFormula] = None) -> bool:
        self._check_existence(cells)
        return self.two_tables.get(cells).satisfiable(evidences)

    @functools.lru_cache(maxsize=None)
    def get_two_tables(self, cells: Tuple[Cell, Cell],
                       evidences: FrozenSet[AtomicFormula] = None) \
            -> Tuple[FrozenSet[AtomicFormula], RingElement]:
        self._check_existence(cells)
        return self.two_tables.get(cells).get_two_tables(evidences)

    def _build_cells(self):
        cells = []
        code = {}
        for model in self.gnd_formula_cc.models():
            for lit in model:
                code[lit.pred] = lit.positive
            cells.append(Cell(tuple(code[p] for p in self.preds), self.preds))
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

    def _build_two_tables(self, gnd_formula_ab: QFFormula):
        # build a pd.DataFrame containing all model as well as the weight
        models = dict()
        gnd_lits = gnd_formula_ab.atoms()
        gnd_lits = gnd_lits.union(
            frozenset(map(lambda x: ~x, gnd_lits))
        )
        for model in gnd_formula_ab.models():
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
                if i > j and self.leq_pred is None:
                    tables[(cell, other_cell)] = tables[(other_cell, cell)]
                    continue
                models_2 = conditional_on(models_1, gnd_lits,
                                          other_cell.get_evidences(b))
                tables[(cell, other_cell)] = TwoTable(
                    models_2, gnd_lits
                )
        return tables


class OptimizedCellGraph(CellGraph):
    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], Tuple[RingElement, RingElement]],
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

    def build_symmetric_cliques(self) -> List[List[Cell]]:
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
            for j in range(self.domain_size):
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


def build_cell_graphs(formula: QFFormula,
                      get_weight: Callable[[Pred],
                                           Tuple[RingElement, RingElement]],
                      leq_pred: Pred = None,
                      predecessor_pred: Pred = None,
                      optimized: bool = False,
                      domain_size: int = 0,
                      modified_cell_symmetry: bool = False) \
        -> Generator[tuple[CellGraph, RingElement]]:
    nullary_atoms = [atom for atom in formula.atoms() if atom.pred.arity == 0]
    if len(nullary_atoms) == 0:
        logger.info('No nullary atoms found, building a single cell graph')
        if not optimized:
            yield CellGraph(
                formula, get_weight, leq_pred, predecessor_pred
            ), Rational(1, 1)
        else:
            yield OptimizedCellGraph(
                formula, get_weight, domain_size, modified_cell_symmetry
            ), Rational(1, 1)
    else:
        logger.info('Found nullary atoms %s', nullary_atoms)
        for values in product(*([[True, False]] * len(nullary_atoms))):
            substitution = dict(zip(nullary_atoms, values))
            logger.info('Building cell graph with values %s', substitution)
            subs_formula = formula.sub_nullary_atoms(substitution).simplify()
            if not subs_formula.satisfiable():
                logger.info('Formula is unsatisfiable, skipping')
                continue
            if not optimized:
                cell_graph = CellGraph(
                    subs_formula, get_weight, leq_pred, predecessor_pred
                )
            else:
                cell_graph = OptimizedCellGraph(
                    subs_formula, get_weight, domain_size,
                    modified_cell_symmetry
                )
            weight = Rational(1, 1)
            for atom, val in zip(nullary_atoms, values):
                weight = weight * (get_weight(atom.pred)[0] if val else get_weight(atom.pred)[1])
            yield cell_graph, weight


class OptimizedCellGraphWithPC(CellGraph):
    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], Tuple[RingElement, RingElement]],
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
            sumn = Rational(0, 1)
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
                                           Tuple[RingElement, RingElement]],
                      leq_pred: Pred = None,
                      predecessor_pred: Pred = None,
                      optimized: bool = False,
                      domain_size: int = 0,
                      modified_cell_symmetry: bool = False,
                      partition_constraint: PartitionConstraint = None) \
        -> Generator[tuple[CellGraph, RingElement]]:
    nullary_atoms = [atom for atom in formula.atoms() if atom.pred.arity == 0]
    if len(nullary_atoms) == 0:
        logger.info('No nullary atoms found, building a single cell graph')
        if not optimized:
            yield CellGraph(
                formula, get_weight, leq_pred, predecessor_pred
            ), Rational(1, 1)
        else:
            if partition_constraint is None:
                yield OptimizedCellGraph(
                    formula, get_weight, domain_size, modified_cell_symmetry
                ), Rational(1, 1)
            else:
                yield OptimizedCellGraphWithPC(
                    formula, get_weight, domain_size, partition_constraint
                ), Rational(1, 1)
    else:
        logger.info('Found nullary atoms %s', nullary_atoms)
        for values in product(*([[True, False]] * len(nullary_atoms))):
            substitution = dict(zip(nullary_atoms, values))
            logger.info('Building cell graph with values %s', substitution)
            subs_formula = formula.sub_nullary_atoms(substitution).simplify()
            if not subs_formula.satisfiable():
                logger.info('Formula is unsatisfiable, skipping')
                continue
            if not optimized:
                cell_graph = CellGraph(
                    subs_formula, get_weight, leq_pred, predecessor_pred
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
