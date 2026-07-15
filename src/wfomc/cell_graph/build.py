"""Build immutable cell-graph data from a typed quantifier-free formula."""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from dataclasses import replace
from functools import reduce
from itertools import product
from time import perf_counter
from typing import TYPE_CHECKING, Generator

from wfomc.arithmetic import ArithmeticValue
from wfomc.fol import (
    Atom,
    Constant as Const,
    Formula as QFFormula,
    Predicate as Pred,
    a,
    atoms,
    b,
    c,
    context_for,
    evaluate,
    free_vars,
    predicates,
    simplify_boolean,
    substitute,
    true,
)
from wfomc.fol.cnf import encode_tseitin
from wfomc.fol.grounding import ground_on_tuple
from .compute_pair_factors import compute_pair_factors
from .data import Cell, CellGraphData, PairFactor
from .enumerate_cells import enumerate_cells, is_satisfiable

if TYPE_CHECKING:
    from wfomc.arithmetic import ArithmeticContext

top = true()
logger = logging.getLogger(__name__)


class _CellGraphBuilder:
    """Single-use builder; only :class:`CellGraphData` escapes this module."""

    def __init__(
        self,
        formula: QFFormula,
        weights: Mapping[Pred, tuple[ArithmeticValue, ArithmeticValue]],
        arithmetic: "ArithmeticContext",
        *,
        predicate_universe: frozenset[Pred],
        leq_pred: Pred | None = None,
        predecessor_preds: dict[int, Pred] | None = None,
        required_unary_preds: frozenset[Pred] = frozenset(),
        cell_formulas: tuple[QFFormula, ...] | None = None,
        projected_binary_preds: tuple[Pred, ...] = (),
    ):
        """Ground one formula branch and enumerate its weighted 1/2-types."""
        build_started = perf_counter()
        self.arithmetic = arithmetic
        for pred in required_unary_preds:
            if pred.arity != 1:
                raise ValueError(f"Required cell predicate {pred} must be unary.")

        self.formula: QFFormula = formula
        self.weights = dict(weights)
        self.leq_pred = leq_pred
        self.predecessor_preds = predecessor_preds
        self.projected_binary_preds = projected_binary_preds
        self.pair_factors_with_preds: dict[int, tuple[tuple[PairFactor, ...], ...]] = {}
        self.cell_formulas = cell_formulas
        self.preds = tuple(
            sorted(
                (predicate for predicate in predicate_universe if predicate.arity > 0),
                key=lambda pred: (pred.name, pred.arity),
            )
        )
        logger.debug("Cell predicates: %s", self.preds)

        ground_started = perf_counter()

        # Both orientations are required because binary predicates need not be
        # symmetric. The diagonal grounding defines cells.
        self.gnd_formula_ab = self._ground_on_tuple(
            self.formula, a, b
        ) & self._ground_on_tuple(self.formula, b, a)
        self.gnd_formula_cc = self._ground_on_tuple(
            self.formula,
            c,
        )
        if self.leq_pred is not None:
            self.gnd_formula_cc &= self.leq_pred(c, c)
            self.gnd_formula_ab = (
                self.gnd_formula_ab & self.leq_pred(b, a) & (~self.leq_pred(a, b))
            )
            if self.predecessor_preds is not None:
                self.gnd_formula_cc = self.gnd_formula_cc & functools.reduce(
                    lambda x, y: x & y,
                    map(lambda x: ~x(c, c), self.predecessor_preds.values()),
                )
                self.gnd_formula_ab_with_preds = dict()
                for idx, predecessor_pred in self.predecessor_preds.items():
                    gnd_formula = (
                        self.gnd_formula_ab
                        & predecessor_pred(b, a)
                        & (~predecessor_pred(a, b))
                    )
                    remainder_preds = list(
                        pred for i, pred in self.predecessor_preds.items() if i != idx
                    )
                    gnd_formula = gnd_formula & functools.reduce(
                        lambda x, y: x & y,
                        map(lambda x: ~x(b, a) & ~x(a, b), remainder_preds),
                        top,
                    )
                    self.gnd_formula_ab_with_preds[idx] = gnd_formula
                self.gnd_formula_ab = self.gnd_formula_ab & functools.reduce(
                    lambda x, y: x & y,
                    map(lambda x: ~x(b, a) & ~x(a, b), self.predecessor_preds.values()),
                )
        ground_ms = (perf_counter() - ground_started) * 1000
        logger.info(
            "Grounded cell-graph branch: predicates=%d pair_atoms=%d "
            "diagonal_atoms=%d predecessor_orders=%s ground_ms=%.3f",
            len(self.preds),
            len(atoms(self.gnd_formula_ab)),
            len(atoms(self.gnd_formula_cc)),
            tuple(sorted((self.predecessor_preds or {}).keys())),
            ground_ms,
        )
        if self.predecessor_preds is not None:
            logger.debug(
                "Grounded predecessor variants: orders=%s",
                tuple(sorted(self.gnd_formula_ab_with_preds)),
            )

        cell_started = perf_counter()
        self.cells = self._build_cells()
        cell_ms = (perf_counter() - cell_started) * 1000

        weight_started = perf_counter()
        self.cell_weights = self._compute_cell_weights()
        weight_ms = (perf_counter() - weight_started) * 1000

        pair_started = perf_counter()
        self.pair_factors = self._build_pair_factors(
            self.gnd_formula_ab,
            phase="base",
        )
        pair_ms = (perf_counter() - pair_started) * 1000

        predecessor_started = perf_counter()
        if self.predecessor_preds is not None:
            self.pair_factors_with_preds = {
                order: self._build_pair_factors(
                    self.gnd_formula_ab_with_preds[order],
                    phase=f"predecessor[{order}]",
                )
                for order in sorted(self.gnd_formula_ab_with_preds)
            }
        predecessor_ms = (perf_counter() - predecessor_started) * 1000
        total_ms = (perf_counter() - build_started) * 1000
        logger.info(
            "Built cell graph: predicates=%d cells=%d pairs=%d "
            "ground_ms=%.3f cell_ms=%.3f weight_ms=%.3f pair_ms=%.3f "
            "predecessor_ms=%.3f total_ms=%.3f",
            len(self.preds),
            len(self.cells),
            len(self.cells) ** 2,
            ground_ms,
            cell_ms,
            weight_ms,
            pair_ms,
            predecessor_ms,
            total_ms,
        )

    def _ground_on_tuple(
        self, formula: QFFormula, c1: Const, c2: Const = None
    ) -> QFFormula:
        """Substitute variables in *formula* with constants ``(c1, c2)``.

        The shared FOL grounding helper also restores the binary atom universe
        needed by pair-model enumeration.
        """
        return ground_on_tuple(formula, c1, c2)

    def snapshot(self) -> CellGraphData:
        """Return the immutable data shared by algorithm preparation code."""

        cells = tuple(self.cells)
        predecessor_factors = tuple(
            (
                order,
                factors,
            )
            for order, factors in sorted(self.pair_factors_with_preds.items())
        )
        return CellGraphData(
            cells=cells,
            arithmetic=self.arithmetic,
            cell_weights=tuple(self.cell_weights[cell] for cell in cells),
            pair_factors=self.pair_factors,
            predecessor_pair_factors=predecessor_factors,
        )

    def _build_cells(self):
        """Build all possible cells (1-types)."""
        cells: list[Cell] = []
        seen: set[Cell] = set()
        cell_formulas = self.cell_formulas
        if cell_formulas is None:
            cell_formulas = (top,)

        for profile_index, cell_formula in enumerate(cell_formulas):
            started = perf_counter()
            gnd_formula = self.gnd_formula_cc
            if cell_formula is not top:
                gnd_formula = gnd_formula & self._ground_on_tuple(cell_formula, c)
            cnf_started = perf_counter()
            cnf = encode_tseitin(gnd_formula)
            cnf_ms = (perf_counter() - cnf_started) * 1000
            candidates = enumerate_cells(cnf, self.preds)
            for cell in candidates:
                if cell in seen:
                    continue
                seen.add(cell)
                cells.append(cell)
            logger.info(
                "Enumerated cells: profile=%d cnf_vars=%d cnf_clauses=%d "
                "candidates=%d unique_total=%d cnf_ms=%.3f total_ms=%.3f",
                profile_index,
                cnf.n_vars,
                len(cnf.clauses),
                len(candidates),
                len(cells),
                cnf_ms,
                (perf_counter() - started) * 1000,
            )
        return cells

    def _compute_cell_weights(self):
        weights = dict()
        for cell in self.cells:
            weight = self.arithmetic.one()
            for i, pred in zip(cell.code, cell.preds):
                assert pred.arity > 0, "Nullary predicates should have been removed"
                if i:
                    weight = self.arithmetic.multiply(
                        weight,
                        self._get_weight(pred)[0],
                    )
                else:
                    weight = self.arithmetic.multiply(
                        weight,
                        self._get_weight(pred)[1],
                    )
            weights[cell] = weight
        return weights

    def _build_pair_factors(
        self,
        gnd_formula_ab: QFFormula,
        *,
        phase: str,
    ) -> tuple[tuple[PairFactor, ...], ...]:
        started = perf_counter()
        if all(predicate.arity == 1 for predicate in self.preds):
            result = self._build_unary_pair_factors(gnd_formula_ab)
            logger.info(
                "Computed unary pair factors: phase=%s cells=%d total_ms=%.3f",
                phase,
                len(self.cells),
                (perf_counter() - started) * 1000,
            )
            return result

        if self.cell_formulas is not None:
            gnd_formula_ab = self._add_profile_constraints(gnd_formula_ab)
        cnf_started = perf_counter()
        cnf = encode_tseitin(gnd_formula_ab)
        cnf_ms = (perf_counter() - cnf_started) * 1000
        logger.info(
            "Computing pair factors: phase=%s cells=%d cnf_vars=%d "
            "cnf_clauses=%d projected_binary=%d cnf_ms=%.3f",
            phase,
            len(self.cells),
            cnf.n_vars,
            len(cnf.clauses),
            len(self.projected_binary_preds),
            cnf_ms,
        )
        result = compute_pair_factors(
            cnf,
            self.cells,
            self.weights,
            self.arithmetic,
            projected_binary_preds=self.projected_binary_preds,
        )
        logger.info(
            "Computed pair factors: phase=%s cells=%d total_ms=%.3f",
            phase,
            len(self.cells),
            (perf_counter() - started) * 1000,
        )
        return result

    def _build_unary_pair_factors(
        self,
        gnd_formula_ab: QFFormula,
    ) -> tuple[tuple[PairFactor, ...], ...]:
        """Evaluate unary pair constraints under the two complete cell types."""

        rows = []
        for left in self.cells:
            left_evidence = left.get_evidences(a)
            row = []
            for right in self.cells:
                model = frozenset(left_evidence | right.get_evidences(b))
                assignment = {literal.atom: literal.positive for literal in model}
                weight = (
                    self.arithmetic.one()
                    if evaluate(gnd_formula_ab, assignment)
                    else self.arithmetic.zero()
                )
                row.append(PairFactor(weight))
            rows.append(tuple(row))
        return tuple(rows)

    def _get_weight(
        self,
        predicate: Pred,
    ) -> tuple[ArithmeticValue, ArithmeticValue]:
        return _weight_pair(self.weights, predicate, self.arithmetic)

    def _add_profile_constraints(
        self,
        gnd_formula_ab: QFFormula,
    ) -> QFFormula:
        if not self.cell_formulas:
            return gnd_formula_ab

        for const in (a, b):
            grounded = tuple(
                self._ground_on_tuple(cell_formula, const)
                for cell_formula in self.cell_formulas
            )
            profile_constraint = reduce(lambda left, right: left | right, grounded)
            gnd_formula_ab &= profile_constraint
        return gnd_formula_ab


def build_cell_graphs(
    formula: QFFormula,
    weights: Mapping[Pred, tuple[ArithmeticValue, ArithmeticValue]],
    arithmetic: "ArithmeticContext",
    leq_pred: Pred | None = None,
    predecessor_preds: dict[int, Pred] | None = None,
    required_unary_preds: frozenset[Pred] = frozenset(),
    cell_formulas: tuple[QFFormula, ...] | None = None,
    projected_binary_preds: tuple[Pred, ...] = (),
) -> Generator[tuple[CellGraphData, ArithmeticValue]]:
    named_constants = tuple(
        sorted(
            {
                term
                for atom in atoms(formula)
                for term in atom.terms
                if isinstance(term, Const)
            },
            key=str,
        )
    )
    if named_constants:
        raise ValueError(
            "cell-graph construction does not support named constants in the "
            f"formula: {named_constants}"
        )

    predicate_universe = set(predicates(formula))
    predicate_universe.update(required_unary_preds)
    predicate_universe.update(projected_binary_preds)
    if leq_pred is not None:
        predicate_universe.add(leq_pred)
    predicate_universe.update((predecessor_preds or {}).values())
    for cell_formula in cell_formulas or ():
        predicate_universe.update(predicates(cell_formula))
    frozen_predicate_universe = frozenset(predicate_universe)

    nullary_atoms = sorted(
        (atom for atom in atoms(formula) if atom.predicate.arity == 0),
        key=lambda atom: atom.predicate.cache_key_parts(),
    )
    logger.info(
        "Starting cell-graph build: predicates=%d nullary=%d "
        "required_unary=%d projected_binary=%d leq=%s predecessor_orders=%s "
        "cell_profiles=%d",
        len(frozen_predicate_universe),
        len(nullary_atoms),
        len(required_unary_preds),
        len(projected_binary_preds),
        leq_pred is not None,
        tuple(sorted((predecessor_preds or {}).keys())),
        len(cell_formulas or ()),
    )
    if len(nullary_atoms) == 0:
        if not free_vars(formula) and not is_satisfiable(encode_tseitin(formula)):
            return
        logger.debug("No nullary atoms; building one cell-graph branch")
        graph = _CellGraphBuilder(
            formula,
            weights,
            arithmetic,
            predicate_universe=frozen_predicate_universe,
            leq_pred=leq_pred,
            predecessor_preds=predecessor_preds,
            required_unary_preds=required_unary_preds,
            cell_formulas=cell_formulas,
            projected_binary_preds=projected_binary_preds,
        )
        yield graph.snapshot(), arithmetic.one()
    else:
        logger.debug("Nullary branching: atoms=%d", len(nullary_atoms))
        for values in product(*([[True, False]] * len(nullary_atoms))):
            substitution = dict(zip(nullary_atoms, values))
            logger.debug("Building nullary branch: %s", substitution)
            subs_formula = _substitute_nullary(formula, substitution)
            if not is_satisfiable(encode_tseitin(subs_formula)):
                logger.debug("Skipping unsatisfiable nullary branch")
                continue
            builder = _CellGraphBuilder(
                subs_formula,
                weights,
                arithmetic,
                predicate_universe=frozen_predicate_universe,
                leq_pred=leq_pred,
                predecessor_preds=predecessor_preds,
                required_unary_preds=required_unary_preds,
                cell_formulas=cell_formulas,
                projected_binary_preds=projected_binary_preds,
            )
            cell_graph = replace(
                builder.snapshot(),
                nullary_assignments=tuple(
                    (atom.predicate, value)
                    for atom, value in zip(nullary_atoms, values)
                ),
            )
            weight = arithmetic.one()
            for atom, val in zip(nullary_atoms, values):
                pair = _weight_pair(weights, atom.predicate, arithmetic)
                weight = arithmetic.multiply(
                    weight,
                    pair[0] if val else pair[1],
                )
            yield cell_graph, weight


def _weight_pair(
    weights: Mapping[Pred, tuple[ArithmeticValue, ArithmeticValue]],
    predicate: Pred,
    arithmetic: "ArithmeticContext",
) -> tuple[ArithmeticValue, ArithmeticValue]:
    pair = weights.get(predicate)
    if pair is not None:
        return pair
    key = (predicate.name, predicate.arity)
    for candidate, candidate_pair in weights.items():
        if (candidate.name, candidate.arity) == key:
            return candidate_pair
    return arithmetic.one(), arithmetic.one()


def _substitute_nullary(
    formula: QFFormula,
    assignment: dict[Atom, bool],
) -> QFFormula:
    ctx = context_for(formula)
    replacement = {
        atom: ctx.true() if value else ctx.false() for atom, value in assignment.items()
    }
    return simplify_boolean(substitute(formula, replacement))
