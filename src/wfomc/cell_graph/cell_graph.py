from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import functools
import networkx as nx
from itertools import product
from typing import Callable, Generator, TYPE_CHECKING
from loguru import logger
from copy import deepcopy

from flint import fmpq as Rational

from wfomc.fol import (
    AtomicFormula,
    Const,
    Pred,
    QFFormula,
    X,
    a,
    b,
    c,
    exactly_one_qf,
    top,
)
from wfomc.utils import RingElement, MultinomialCoefficients
from .components import Cell, TwoTable
from .utils import conditional_on

if TYPE_CHECKING:
    from wfomc.context.unary_evidence import UnaryEvidencePartition

"""
cell_graph.py is the core data structure module of the WFOMC (Weighted First-Order Model Counting) algorithm. Its main function is to convert an abstract first-order logic problem into a specific and computable graph structure, namely "cell graph".

This process is a crucial step in "Lifted Inference", and its underlying concept is as follows:

Cell: In the domain of discourse, all elements that share the same property (satisfy the same unary predicate) are grouped together. This group is called a "cell" or a "1-type".
Cell Graph:
    The nodes of the graph represent all possible cells.
    Each node has a weight, indicating the weight of a single element belonging to that cell.
    The edges of the graph represent the interaction relationships between two cells. The weight of the edge (represented by the TwoTable in the code) describes the weighted model number of the binary predicate satisfied when one element is taken from each of these two cells.
    By constructing this graph, the algorithm can elevate reasoning from the level of individual elements to the level of "cell types", thereby efficiently handling large domain problems.

The file mainly contains three core classes and a factory function:
    CellGraph: The basic implementation of the cell graph.
    OptimizedCellGraph: An optimized version of CellGraph that accelerates computation by identifying and exploiting symmetries (cliques) and independencies, used in the FastWFOMC algorithm.
    OptimizedCellGraphWithEvidence: Expands OptimizedCellGraph by evidence profiles for lifted unary evidence.
    build_cell_graphs: A factory function that decides which type of CellGraph instance to create based on input parameters.
"""
class CellGraph(object):
    """
    Cell graph that handles cells and the wmc between them.
    """

    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                 leq_pred: Pred = None,
                 predecessor_preds: dict[int, Pred] = None,
                 required_unary_preds: frozenset[Pred] = frozenset(),
                 cell_formulas: tuple[QFFormula, ...] | None = None):
        """
        Cell graph that handles cells (1-types) and the WMC between them

        :param sentence QFFormula: the sentence in the form of quantifier-free formula
        :param get_weight Callable[[Pred], tuple[RingElement, RingElement]]: the weighting function
        :param conditional_formulas list[CNF]: the optional conditional formula appended in WMC computing
        """
        self.required_unary_preds = tuple(sorted(
            required_unary_preds, key=lambda pred: (pred.name, pred.arity)
        ))
        for pred in self.required_unary_preds:
            if pred.arity != 1:
                raise ValueError(
                    f"Required cell predicate {pred} must be unary."
                )
            if pred not in formula.preds():
                formula &= pred(X) | ~pred(X)

        self.formula: QFFormula = formula
        self.get_weight: Callable[[Pred],
                                  tuple[RingElement, RingElement]] = get_weight
        self.leq_pred: Pred = leq_pred
        self.predecessor_preds: dict[int, Pred] = predecessor_preds
        self.cell_formulas = cell_formulas
        self.preds: tuple[Pred] = tuple(self.formula.preds())
        logger.debug('prednames: {}', self.preds)

        # --- Instantiate formulas ---
        gnd_formula_ab1: QFFormula = self._ground_on_tuple(
            self.formula, a, b
        ) # To compute cells (1-types) and their relationships (2-types), the formula needs to be instantiated. # Replace variables in the formula with (a, b)
        gnd_formula_ab2: QFFormula = self._ground_on_tuple(
            self.formula, b, a
        )# Replace variables in the formula with (b, a) to ensure symmetry
        self.gnd_formula_ab: QFFormula = \
            gnd_formula_ab1 & gnd_formula_ab2 # `gnd_formula_ab` is used to calculate the interaction between any two elements and is the conjunction of the two.
        self.gnd_formula_cc: QFFormula = self._ground_on_tuple(
            self.formula, c
        )# `gnd_formula_cc` is used to define the type of a single element (cell), replacing variables with (c, c).
        if self.leq_pred is not None: # If a linear order exists, the order relation needs to be added to the instantiated formula.
            self.gnd_formula_cc = self.gnd_formula_cc & self.leq_pred(c, c) # A single element must satisfy c <= c (reflexivity).
            self.gnd_formula_ab = self.gnd_formula_ab & \
                self.leq_pred(b, a) & \
                (~self.leq_pred(a, b))
            if self.predecessor_preds is not None:
                self.gnd_formula_cc = self.gnd_formula_cc & \
                    functools.reduce(
                        lambda x, y: x & y,
                        map(lambda x: ~x(c, c),
                            self.predecessor_preds.values())
                    )
                self.gnd_formula_ab_with_preds = dict()
                for idx, predecessor_pred in self.predecessor_preds.items():
                    gnd_formula = self.gnd_formula_ab & \
                        predecessor_pred(b, a) & \
                        (~predecessor_pred(a, b))
                    remainder_preds = list(
                        pred for i, pred in self.predecessor_preds.items()
                        if i != idx
                    )
                    gnd_formula = gnd_formula & \
                        functools.reduce(
                            lambda x, y: x & y,
                            map(lambda x: ~x(b, a) & ~x(a, b),
                                remainder_preds),
                            top
                        )
                    self.gnd_formula_ab_with_preds[idx] = gnd_formula
                self.gnd_formula_ab = self.gnd_formula_ab & \
                    functools.reduce(
                        lambda x, y: x & y,
                        map(lambda x: ~x(b, a) & ~x(a, b),
                            self.predecessor_preds.values())
                    )
        logger.info('ground a b: {}', self.gnd_formula_ab)
        logger.info('ground c: {}', self.gnd_formula_cc)
        if self.predecessor_preds is not None:
            logger.info(f'ground a, b with predecessor: {self.gnd_formula_ab_with_preds}')

        # build cells
        self.cells: list[Cell] = self._build_cells() # `_build_cells` finds all possible cell types based on `gnd_formula_cc`.
        # filter cells
        logger.info('the number of valid cells: {}',
                    len(self.cells))

        logger.info('computing cell weights')
        self.cell_weights: dict[Cell, RingElement] = self._compute_cell_weights() # `_compute_cell_weights` computes the weight of each cell.
        logger.info('computing two table weights')
        self.two_tables: dict[tuple[Cell, Cell],
                              TwoTable] = self._build_two_tables(self.gnd_formula_ab)
        if self.predecessor_preds is not None:
            self.two_tables_with_preds: dict[
                int, dict[tuple[Cell, Cell], TwoTable]
            ] = dict(
                (idx, self._build_two_tables(gnd_formula))
                for idx, gnd_formula in self.gnd_formula_ab_with_preds.items()
            )

    def _ground_on_tuple(self, formula: QFFormula,
                         c1: Const, c2: Const = None) -> QFFormula:
        """Helper function to substitute variables in the formula with constants (c1, c2)."""
        variables = formula.vars() # Extract all free variables from the input formula. For example, for the formula P(x) & R(x, y), `variables` will be {'x', 'y'}.
        if len(variables) > 2:
            raise RuntimeError(
                "Can only ground out FO2"
            )
        if len(variables) == 1: # Case 1: The formula has only one variable, e.g., P(x).
            constants = [c1] # Then this variable will be replaced by the first constant `c1`.
        else: # Case 2: The formula has two or zero variables.
            if c2 is not None: # If the second constant `c2` is provided (i.e., not None).
                constants = [c1, c2]
            else: # If the second constant `c2` is not provided.
                constants = [c1, c1]
        substitution = dict(zip(variables, constants)) # Create a substitution dictionary. `zip` pairs the list of variables with the list of constants. For example, if variables are (x, y) and constants are [a, b], the dictionary will be {'x': a, 'y': b}.
        gnd_formula = formula.substitute(substitution) # Call the `substitute` method of the formula object, passing the substitution dictionary. This method returns a new formula object with variables replaced by constants.
        # NOTE: workaround for the case where ground binary atoms not appearing in the formula
        if c2 is not None:
            binary_preds = list(filter(
                lambda x: x.arity == 2, gnd_formula.preds()
            ))
            for pred in binary_preds:
                atom = pred(c1, c2)
                if atom not in gnd_formula.atoms():
                    gnd_formula = gnd_formula & (atom | ~atom)
            # NOTE: end workaround
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
        """Obtain the list of cells, and selectively apply filtering."""
        if cell_filter is None:
            return self.cells
        return list(filter(cell_filter, self.cells))

    @functools.lru_cache(maxsize=None, typed=True)
    def get_cell_weight(self, cell: Cell) -> RingElement:
        """Get the weight of a single cell (with caching)."""
        if cell not in self.cell_weights:
            logger.warning(
                "Cell %s not found", cell
            )
            return 0
        return self.cell_weights.get(cell)

    def _check_existence(self, cells: tuple[Cell, Cell]):
        """Check if the given pair of cells exists in two_tables."""
        if cells not in self.two_tables:
            raise ValueError(
                f"Cells {cells} not found, note that the order of cells matters!"
            )

    @functools.lru_cache(maxsize=None, typed=True)
    def get_two_table_weight(self, cells: tuple[Cell, Cell],
                             evidences: frozenset[AtomicFormula] = None) -> RingElement:
        """Obtain the interaction weight between two cells (with caching)."""
        self._check_existence(cells)
        return self.two_tables.get(cells).get_weight(evidences)

    @functools.lru_cache(maxsize=None, typed=True)
    def get_two_table_with_pred_weight(
        self, cells: tuple[Cell, Cell], index: int,
        evidences: frozenset[AtomicFormula] = None
    ) -> RingElement:
        self._check_existence(cells)
        return self.two_tables_with_preds[index].get(cells).get_weight(evidences)

    def get_all_weights(self) -> tuple[list[RingElement], list[list[RingElement]]]:
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
            -> tuple[tuple[frozenset[AtomicFormula], RingElement], ...]:
        self._check_existence(cells)
        return self.two_tables.get(cells).get_two_tables(evidences)

    def _build_cells(self):
        """Build all possible cells (1-types)."""
        cells = []
        seen = set()
        cell_formulas = self.cell_formulas
        if cell_formulas is None:
            cell_formulas = (top,)

        for cell_formula in cell_formulas:
            gnd_formula = self.gnd_formula_cc
            if cell_formula is not top:
                gnd_formula = gnd_formula & self._ground_on_tuple(cell_formula, c)
            for model in gnd_formula.models(): # Iterate over all models of the single-element grounded formula `gnd_formula_cc`. Each model represents a valid element type.
                code = {}
                for lit in model: # Convert the model (a set of truth assignments) into a code.
                    code[lit.pred] = lit.positive
                cell = Cell(tuple(code[p] for p in self.preds), self.preds) # Create a Cell object with this code.
                if cell in seen:
                    continue
                seen.add(cell)
                cells.append(cell)
        return cells

    def _compute_cell_weights(self):
        weights = dict()
        for cell in self.cells:
            weight = 1
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
        weight = 1
        for i, pred in zip(cell.code, cell.preds):
            if pred.arity == 0:
                if i:
                    weight = weight * self.get_weight(pred)[0]
                else:
                    weight = weight * self.get_weight(pred)[1]
        return weight

    def _build_two_tables(self, gnd_formula_ab: QFFormula):
        # build a pd.DataFrame containing all model as well as the weight
        if all(pred.arity == 1 for pred in gnd_formula_ab.preds()):
            gnd_lits = gnd_formula_ab.atoms()
            gnd_lits = gnd_lits.union(
                frozenset(map(lambda x: ~x, gnd_lits))
            )
            tables = dict()
            for cell in self.cells:
                cell_evidences = cell.get_evidences(a)
                for other_cell in self.cells:
                    model = frozenset(
                        cell_evidences | other_cell.get_evidences(b)
                    )
                    tables[(cell, other_cell)] = TwoTable(
                        {model: Rational(1, 1)}, gnd_lits
                    )
            return tables

        internal_predicates = frozenset()
        if self.cell_formulas is not None:
            gnd_formula_ab, internal_predicates = (
                self._add_profile_selectors(gnd_formula_ab)
            )

        models = dict()
        gnd_lits = frozenset(
            atom
            for atom in gnd_formula_ab.atoms()
            if atom.pred not in internal_predicates
        )
        gnd_lits = gnd_lits.union(
            frozenset(map(lambda x: ~x, gnd_lits))
        )
        for model in gnd_formula_ab.models():
            model = frozenset(
                lit
                for lit in model
                if lit.pred not in internal_predicates
            )
            weight = 1
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
                # if i > j and self.leq_pred is None: # When leq_pred is not defined, the relationship between (cell, other_cell) and (other_cell, cell) is considered undefined. Because when i > j, the tuple (other_cell, cell) corresponding to index j and index i has already been calculated in the previous loop.
                #     tables[(cell, other_cell)] = tables[(other_cell, cell)] # The constraint on B(X, Y) is about "out-degree". It requires that the "out-degree" of each node X (the number of Y for which B(X, Y) is true) must be odd. However, it imposes no requirements on the "in-degree" of the node (the number of Y for which B(Y, X) is true). Therefore, the truth values of B(X, Y) and B(Y, X) may be different.
                models_2 = conditional_on(models_1, gnd_lits,
                                          other_cell.get_evidences(b))
                tables[(cell, other_cell)] = TwoTable(
                    models_2, gnd_lits
                )
        return tables

    def _add_profile_selectors(
        self,
        gnd_formula_ab: QFFormula,
    ) -> tuple[QFFormula, frozenset[Pred]]:
        if not self.cell_formulas:
            return gnd_formula_ab, frozenset()

        selector_predicates: set[Pred] = set()
        for const in (a, b):
            selector_formula, predicates = self._profile_selector_formula(
                const
            )
            gnd_formula_ab = gnd_formula_ab & selector_formula
            selector_predicates.update(predicates)
        return gnd_formula_ab, frozenset(selector_predicates)

    def _profile_selector_formula(
        self,
        const: Const,
    ) -> tuple[QFFormula, frozenset[Pred]]:
        if len(self.cell_formulas) == 1:
            formula = self._ground_on_tuple(self.cell_formulas[0], const)
            return formula, frozenset()

        selector_preds = tuple(
            Pred(f"@cell_profile_{const.name}_{idx}", 1)
            for idx, _ in enumerate(self.cell_formulas)
        )
        formula = self._ground_on_tuple(
            exactly_one_qf(list(selector_preds)), const
        )
        for selector_pred, cell_formula in zip(
            selector_preds, self.cell_formulas
        ):
            formula = formula & selector_pred(const).implies(
                self._ground_on_tuple(cell_formula, const)
            )
        return formula, frozenset(selector_preds)


class OptimizedCellGraph(CellGraph):
    def __init__(self, formula: QFFormula,
                 get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
                 domain_size: int,
                 modified_cell_symmetry: bool = False,
                 required_unary_preds: frozenset[Pred] = frozenset()):
        """
        Optimized cell graph for FastWFOMC
        :param formula: the formula to be grounded
        :param get_weight: a function that returns the weight of a predicate
        :param domain_size: the domain size
        """
        super().__init__(
            formula, get_weight, required_unary_preds=required_unary_preds
        )
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

        logger.info("Found i1 independent cliques: {}", self.i1_ind)
        logger.info("Found i2 independent cliques: {}", self.i2_ind)
        logger.info("Found non-independent cliques: {}", self.nonind)

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
        logger.info("Built {} symmetric cliques: {}", len(cliques), cliques)
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
        logger.info("Built {} symmetric cliques: {}", len(cliques), cliques)
        return cliques, ind_idx

    def find_independent_sets(self) -> tuple[list[int], list[int], list[int], list[int]]:
        g = nx.Graph()
        g.add_nodes_from(range(len(self.cells)))
        for i in range(len(self.cells)):
            for j in range(i + 1, len(self.cells)):
                if self.get_two_table_weight(
                        (self.cells[i], self.cells[j])
                ) != 1:
                    g.add_edge(i, j)

        self_loop = set()
        for i in range(len(self.cells)):
            if self.get_two_table_weight((self.cells[i], self.cells[i])) != 1:
                self_loop.add(i)

        non_self_loop = g.nodes - self_loop
        if len(non_self_loop) == 0:
            i1_ind = set()
        else:
            i1_ind = set(nx.maximal_independent_set(g.subgraph(non_self_loop)))
        g_ind = set(nx.maximal_independent_set(g, nodes=i1_ind))
        i2_ind = g_ind.difference(i1_ind)
        non_ind = g.nodes - i1_ind - i2_ind
        logger.info("Found i1 independent set: {}", i1_ind)
        logger.info("Found i2 independent set: {}", i2_ind)
        logger.info("Found non-independent set: {}", non_ind)
        return list(i1_ind), list(i2_ind), list(non_ind)

    def find_independent_cliques(self) -> tuple[list[int], list[int], list[int], list[int]]:
        g = nx.Graph()
        g.add_nodes_from(range(len(self.cliques)))
        for i in range(len(self.cliques)):
            for j in range(i + 1, len(self.cliques)):
                if self.get_two_table_weight(
                        (self.cliques[i][0], self.cliques[j][0])
                ) != 1:
                    g.add_edge(i, j)

        self_loop = set()
        for i in range(len(self.cliques)):
            for j in range(1, self.domain_size + 1):
                if self.get_J_term(i, j) != 1:
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
            accum = 0
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


@dataclass(frozen=True)
class CellWithEvidenceProfile:
    """A lightweight compatible ``(base cell, evidence profile)`` node."""

    base_cell: Cell
    evidence_profile_index: int

    def is_positive(self, pred: Pred) -> bool:
        return self.base_cell.is_positive(pred)

    def __str__(self) -> str:
        return f"{self.base_cell} [evidence_profile={self.evidence_profile_index}]"

    def __repr__(self) -> str:
        return str(self)


class OptimizedCellGraphWithEvidence:
    """Fastv2 graph view expanded by evidence profiles without aux predicates."""

    def __init__(
        self,
        formula: QFFormula,
        get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
        domain_size: int,
        unary_evidence_partition: "UnaryEvidencePartition",
        required_unary_preds: frozenset[Pred] = frozenset(),
    ):
        self.base_graph = CellGraph(
            formula,
            get_weight,
            required_unary_preds=required_unary_preds,
            cell_formulas=_cell_formulas_from_evidence_partition(
                unary_evidence_partition
            ),
        )
        self.formula = self.base_graph.formula
        self.preds = self.base_graph.preds
        self.domain_size = domain_size
        self.unary_evidence_partition = unary_evidence_partition
        base_cells = tuple(self.base_graph.get_cells())
        self.allocation = unary_evidence_partition.compile_for_cells(base_cells)
        self.evidence_profile_sizes = self.allocation.evidence_profile_sizes
        self.cells = [
            CellWithEvidenceProfile(base_cells[cell_idx], evidence_profile_idx)
            for cell_idx, evidence_profile_indices in enumerate(
                self.allocation.compatible_evidence_profiles_by_cell
            )
            for evidence_profile_idx in evidence_profile_indices
        ]

        if not self.cells:
            self.i1_ind = []
            self.cliques = []
            self.nonind = []
            self.nonind_map = {}
            self.clique_evidence_profile_partitions = {}
            self.evidence_profile_cliques = defaultdict(list)
            self.i1_evidence_profile_partition = [
                [] for _ in range(len(self.evidence_profile_sizes))
            ]
            logger.info("No evidence-compatible cells found")
            return

        i1_ind_set, i2_ind_set, nonind_set = self.find_independent_sets()
        self.i1_ind = i1_ind_set
        nonind_set = i2_ind_set + nonind_set
        self.cliques, self.nonind = self.build_symmetric_cliques(nonind_set)
        self.nonind_map = dict(zip(self.nonind, range(len(self.nonind))))

        self.clique_evidence_profile_partitions: dict[int, list[list[int]]] = {}
        self.evidence_profile_cliques: dict[int, list[int]] = defaultdict(list)
        for clique_idx, clique in enumerate(self.cliques):
            clique_evidence_profile_partition = []
            for evidence_profile_idx in range(len(self.evidence_profile_sizes)):
                cell_indices = [
                    cell_idx
                    for cell_idx, cell in enumerate(clique)
                    if cell.evidence_profile_index == evidence_profile_idx
                ]
                if cell_indices:
                    clique_evidence_profile_partition.append(cell_indices)
                    self.evidence_profile_cliques[evidence_profile_idx].append(
                        clique_idx
                    )
            self.clique_evidence_profile_partitions[clique_idx] = (
                clique_evidence_profile_partition
            )

        self.i1_evidence_profile_partition = [
            [
                cell_idx
                for cell_idx in self.i1_ind
                if self.cells[cell_idx].evidence_profile_index == evidence_profile_idx
            ]
            for evidence_profile_idx in range(len(self.evidence_profile_sizes))
        ]

        logger.info(
            "Cell-evidence compatibility pairs: {}",
            self.allocation.compatibility_pair_count,
        )
        logger.info("Evidence-expanded cells: {}", len(self.cells))
        logger.info(
            "Evidence clique partitions: {}",
            self.clique_evidence_profile_partitions,
        )
        logger.info("Evidence cliques: {}", self.evidence_profile_cliques)

    def get_cells(self):
        return self.cells

    @functools.lru_cache(maxsize=None, typed=True)
    def get_cell_weight(self, cell: CellWithEvidenceProfile) -> RingElement:
        return self.base_graph.get_cell_weight(cell.base_cell)

    @functools.lru_cache(maxsize=None, typed=True)
    def get_two_table_weight(
        self,
        cells: tuple[CellWithEvidenceProfile, CellWithEvidenceProfile],
    ) -> RingElement:
        return self.base_graph.get_two_table_weight(
            (cells[0].base_cell, cells[1].base_cell)
        )

    def find_independent_sets(self) -> tuple[list[int], list[int], list[int]]:
        max_evidence_profile_idx = max(
            range(len(self.evidence_profile_sizes)),
            key=lambda idx: self.evidence_profile_sizes[idx],
        )
        in_max_evidence_profile = {
            idx
            for idx, cell in enumerate(self.cells)
            if cell.evidence_profile_index == max_evidence_profile_idx
        }

        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.cells)))
        for i in range(len(self.cells)):
            for j in range(i + 1, len(self.cells)):
                if self.get_two_table_weight(
                    (self.cells[i], self.cells[j])
                ) != Rational(1, 1):
                    graph.add_edge(i, j)

        self_loop = {
            i
            for i in range(len(self.cells))
            if self.get_two_table_weight(
                (self.cells[i], self.cells[i])
            ) != Rational(1, 1)
        }
        non_self_loop = set(graph.nodes) - self_loop
        if not non_self_loop:
            i1_ind = set()
        elif non_self_loop & in_max_evidence_profile:
            evidence_seed = set(nx.maximal_independent_set(
                graph.subgraph(non_self_loop & in_max_evidence_profile)
            ))
            i1_ind = set(nx.maximal_independent_set(
                graph.subgraph(non_self_loop), nodes=evidence_seed
            ))
        else:
            i1_ind = set(nx.maximal_independent_set(
                graph.subgraph(non_self_loop)
            ))

        graph_ind = set(nx.maximal_independent_set(graph, nodes=i1_ind))
        i2_ind = graph_ind - i1_ind
        nonind = set(graph.nodes) - i1_ind - i2_ind
        logger.info("Found evidence i1 independent set: {}", i1_ind)
        logger.info("Found evidence i2 independent set: {}", i2_ind)
        logger.info("Found evidence non-independent set: {}", nonind)
        return list(i1_ind), list(i2_ind), list(nonind)

    def _matches(self, clique, other_cell) -> bool:
        cell = clique[0]
        if len(clique) > 1:
            relation = self.get_two_table_weight((cell, clique[1]))
            for third_cell in clique:
                if relation != self.get_two_table_weight(
                    (other_cell, third_cell)
                ):
                    return False

        for third_cell in self.get_cells():
            if other_cell == third_cell or third_cell in clique:
                continue
            relation = self.get_two_table_weight((cell, third_cell))
            if relation != self.get_two_table_weight(
                (other_cell, third_cell)
            ):
                return False
        return True

    def build_symmetric_cliques(
        self,
        cell_indices: list[int],
    ) -> tuple[list[list[CellWithEvidenceProfile]], list[int]]:
        cell_indices = list(cell_indices)
        cliques = []
        clique_indices = []
        while cell_indices:
            cell_idx = cell_indices.pop()
            clique = [self.cells[cell_idx]]
            for other_cell_idx in cell_indices:
                other_cell = self.cells[other_cell_idx]
                if self._matches(clique, other_cell):
                    clique.append(other_cell)
            for other_cell in clique[1:]:
                cell_indices.remove(self.cells.index(other_cell))
            cliques.append(clique)
            clique_indices.append(len(cliques) - 1)
        logger.info(
            "Built {} evidence symmetric cliques: {}",
            len(cliques),
            cliques,
        )
        return cliques, clique_indices

    def get_i1_weight(self, i1_config: tuple[int], config: tuple[int]) -> RingElement:
        result = Rational(1, 1)
        for i1_indices, count in zip(
            self.i1_evidence_profile_partition, i1_config
        ):
            if not i1_indices and count > 0:
                return Rational(0, 1)
            accum = Rational(0, 1)
            for cell_idx in i1_indices:
                term = self.get_cell_weight(self.cells[cell_idx])
                for clique_idx in self.nonind:
                    term *= self.get_two_table_weight(
                        (self.cliques[clique_idx][0], self.cells[cell_idx])
                    ) ** config[self.nonind_map[clique_idx]]
                accum += term
            result *= accum ** count
        return result

    @functools.lru_cache(maxsize=None)
    def get_J_term(self, clique_idx: int, clique_config: tuple[int]) -> RingElement:
        result = Rational(1, 1)
        clique = self.cliques[clique_idx]
        evidence_profile_groups = self.clique_evidence_profile_partitions[
            clique_idx
        ]
        if len(evidence_profile_groups) == 1:
            return self.get_partitioned_J_term(
                clique_idx, 0, clique_config[0]
            )

        relation = self.get_two_table_weight((clique[0], clique[1]))
        cross_pairs = sum(
            left_count * right_count
            for left_idx, left_count in enumerate(clique_config)
            for right_idx, right_count in enumerate(clique_config)
            if left_idx < right_idx
        )
        result *= relation ** cross_pairs
        for partition_idx in range(len(evidence_profile_groups)):
            result *= self.get_partitioned_J_term(
                clique_idx, partition_idx, clique_config[partition_idx]
            )
        return result

    @functools.lru_cache(maxsize=None)
    def get_partitioned_J_term(
        self,
        clique_idx: int,
        partition_idx: int,
        count: int,
    ) -> RingElement:
        cell_indices = self.clique_evidence_profile_partitions[clique_idx][
            partition_idx
        ]
        clique = self.cliques[clique_idx]
        if len(cell_indices) == 1:
            cell = clique[cell_indices[0]]
            return (
                self.get_two_table_weight((cell, cell))
                ** MultinomialCoefficients.comb(count, 2)
                * self.get_cell_weight(cell) ** count
            )
        return self.get_d_term(clique_idx, count, partition_idx)

    @functools.lru_cache(maxsize=None)
    def get_d_term(
        self,
        clique_idx: int,
        count: int,
        partition_idx: int,
        cur: int = 0,
    ) -> RingElement:
        cell_indices = self.clique_evidence_profile_partitions[clique_idx][
            partition_idx
        ]
        cell_index = cell_indices[cur]
        clique = self.cliques[clique_idx]
        cell = clique[cell_index]
        relation = self.get_two_table_weight((clique[0], clique[1]))
        self_relation = self.get_two_table_weight((cell, cell))
        weight = self.get_cell_weight(cell)

        if cur == len(cell_indices) - 1:
            return (
                weight ** count
                * self_relation ** MultinomialCoefficients.comb(count, 2)
            )

        result = Rational(0, 1)
        for cell_count in range(count + 1):
            term = MultinomialCoefficients.comb(count, cell_count)
            term *= weight ** cell_count
            term *= self_relation ** MultinomialCoefficients.comb(cell_count, 2)
            term *= relation ** (cell_count * (count - cell_count))
            term *= self.get_d_term(
                clique_idx,
                count - cell_count,
                partition_idx,
                cur + 1,
            )
            result += term
        return result


def _cell_formulas_from_evidence_partition(
    unary_evidence_partition: "UnaryEvidencePartition | None",
) -> tuple[QFFormula, ...] | None:
    if (
        unary_evidence_partition is None
        or not unary_evidence_partition.covers_all_elements
    ):
        return None

    return tuple(
        functools.reduce(
            lambda left, right: left & right,
            evidence_profile.evidence,
            top,
        )
        for evidence_profile in unary_evidence_partition.evidence_profiles
    )


def build_cell_graphs(
    formula: QFFormula,
    get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
    leq_pred: Pred = None,
    predecessor_preds: dict[int, Pred] = None,
    optimized: bool = False,
    domain_size: int = 0,
    modified_cell_symmetry: bool = False,
    required_unary_preds: frozenset[Pred] = frozenset(),
    unary_evidence_partition: "UnaryEvidencePartition | None" = None,
) -> Generator[tuple[CellGraph, RingElement]]:
    cell_formulas = _cell_formulas_from_evidence_partition(
        unary_evidence_partition
    )
    nullary_atoms = [atom for atom in formula.atoms() if atom.pred.arity == 0]
    if len(nullary_atoms) == 0:
        logger.info('No nullary atoms found, building a single cell graph')
        if not optimized: # Decide whether to create the basic version or the optimized version based on the `optimized` parameter.
            yield CellGraph(
                formula, get_weight, leq_pred, predecessor_preds,
                required_unary_preds,
                cell_formulas=cell_formulas,
            ), Rational(1, 1)
        else:
            if unary_evidence_partition is not None:
                yield OptimizedCellGraphWithEvidence(
                    formula, get_weight, domain_size,
                    unary_evidence_partition, required_unary_preds
                ), Rational(1, 1)
            else:
                yield OptimizedCellGraph(
                    formula, get_weight, domain_size, modified_cell_symmetry,
                    required_unary_preds
                ), Rational(1, 1)
    else:
        logger.info('Found nullary atoms {}', nullary_atoms)
        for values in product(*([[True, False]] * len(nullary_atoms))):
            substitution = dict(zip(nullary_atoms, values))
            logger.info('Building cell graph with values {}', substitution)
            subs_formula = formula.sub_nullary_atoms(substitution).simplify()
            if not subs_formula.satisfiable():
                logger.info('Formula is unsatisfiable, skipping')
                continue
            if not optimized:
                cell_graph = CellGraph(
                    subs_formula, get_weight, leq_pred, predecessor_preds,
                    required_unary_preds,
                    cell_formulas=cell_formulas,
                )
            else:
                if unary_evidence_partition is not None:
                    cell_graph = OptimizedCellGraphWithEvidence(
                        subs_formula, get_weight, domain_size,
                        unary_evidence_partition, required_unary_preds
                    )
                else:
                    cell_graph = OptimizedCellGraph(
                        subs_formula, get_weight, domain_size,
                        modified_cell_symmetry, required_unary_preds
                    )
            weight = Rational(1, 1)
            for atom, val in zip(nullary_atoms, values):
                weight = weight * (get_weight(atom.pred)[0] if val else get_weight(atom.pred)[1])
            yield cell_graph, weight
