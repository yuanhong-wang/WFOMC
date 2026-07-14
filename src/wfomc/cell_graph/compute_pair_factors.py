"""Compute pair factors with bounded PySAT, Ganak, and PySDD backup."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from time import perf_counter

from pysat.solvers import Solver
from pysdd.sdd import SddManager, Vtree

from wfomc.arithmetic import ArithmeticContext, ArithmeticValue
from wfomc.errors import GanakError
from wfomc.fol import Atom, Predicate, a, b
from wfomc.fol.cnf import TseitinCNF

from .compute_pair_factors_ganak import compute_factor_with_ganak
from .data import Cell, PairFactor


Factor = dict[int, ArithmeticValue]
WeightPair = tuple[ArithmeticValue, ArithmeticValue]
logger = logging.getLogger(__name__)


# Small local pair theories are cheaper to enumerate than to compile to an
# SDD.  The total-model cap keeps this a bounded fast path; large theories
# retain the existing one-traversal PySDD backend.
_PYSAT_MODEL_LIMIT = 128
_PYSAT_CONDITION_LIMIT = 64
_GANAK_TIMEOUT_SECONDS = 30.0


def compute_pair_factors(
    cnf: TseitinCNF,
    cells: Sequence[Cell],
    predicate_weights: Mapping[Predicate, WeightPair],
    arithmetic: ArithmeticContext,
    *,
    projected_binary_preds: Sequence[Predicate] = (),
) -> tuple[tuple[PairFactor, ...], ...]:
    """Build conditioned pair factors through the shared backend cascade."""

    started = perf_counter()
    cell_tuple = tuple(cells)
    if not cell_tuple:
        return ()
    projected_preds = tuple(projected_binary_preds)
    for predicate in projected_preds:
        if predicate.arity != 2:
            raise ValueError(f"Projected pair predicate {predicate} must be binary")

    condition_atoms = _condition_atoms(cnf, cell_tuple[0].preds)
    condition_masks = tuple(
        sorted(
            {
                _condition_mask(condition_atoms, left, right)
                for left in cell_tuple
                for right in cell_tuple
            }
        )
    )
    counting_projection = _counting_projection(cnf, projected_preds)
    counting_bit_count = 2 * len(projected_preds)
    condition_projection = {
        cnf.atom_to_var[atom]: counting_bit_count + index
        for index, atom in enumerate(condition_atoms)
    }
    combined_projection = dict(counting_projection)
    combined_projection.update(condition_projection)
    literal_weights = _literal_weights(
        cnf,
        condition_atoms,
        predicate_weights,
        arithmetic,
    )
    free_factor = _free_offdiagonal_factor(
        cnf,
        cell_tuple[0].preds,
        projected_preds,
        predicate_weights,
        arithmetic,
    )

    prepare_ms = (perf_counter() - started) * 1000
    logger.info(
        "Starting pair-factor computation: cnf_vars=%d cnf_clauses=%d "
        "atoms=%d cells=%d condition_atoms=%d condition_signatures=%d "
        "projected_binary=%d prepare_ms=%.3f",
        cnf.n_vars,
        len(cnf.clauses),
        len(cnf.atoms),
        len(cell_tuple),
        len(condition_atoms),
        len(condition_masks),
        len(projected_preds),
        prepare_ms,
    )

    enumerate_started = perf_counter()
    if len(condition_masks) <= _PYSAT_CONDITION_LIMIT:
        combined, enumerated_models = _enumerate_bounded(
            cnf,
            condition_atoms,
            condition_masks,
            literal_weights,
            combined_projection,
            arithmetic,
            model_limit=_PYSAT_MODEL_LIMIT,
        )
    else:
        combined, enumerated_models = None, 0
    enumerate_ms = (perf_counter() - enumerate_started) * 1000

    backend = "pysat-enum" if combined is not None else ""
    ganak_ms = 0.0
    if combined is None:
        ganak_started = perf_counter()
        try:
            combined = compute_factor_with_ganak(
                cnf,
                literal_weights,
                combined_projection,
                arithmetic,
                timeout=_GANAK_TIMEOUT_SECONDS,
            )
        except GanakError as exc:
            logger.info("Ganak pair-factor backend unavailable: %s", exc)
        ganak_ms = (perf_counter() - ganak_started) * 1000
        if combined is not None:
            backend = "ganak"

    compile_ms = 0.0
    evaluate_ms = 0.0
    if combined is None:
        backend = "pysdd-backup"
        compile_started = perf_counter()
        circuit = _SddCircuit(cnf, arithmetic)
        compile_ms = (perf_counter() - compile_started) * 1000
        try:
            evaluate_started = perf_counter()
            combined = circuit.evaluate(literal_weights, combined_projection)
            evaluate_ms = (perf_counter() - evaluate_started) * 1000
        finally:
            circuit.close()
    combined = _factor_multiply(combined, free_factor, arithmetic)

    materialize_started = perf_counter()
    relation_mask = (1 << counting_bit_count) - 1
    by_condition: dict[int, Factor] = {}
    for mask, value in combined.items():
        condition_mask = mask >> counting_bit_count
        counting_mask = mask & relation_mask
        factor = by_condition.setdefault(condition_mask, {})
        factor[counting_mask] = arithmetic.add(
            factor.get(counting_mask, arithmetic.zero()),
            value,
        )

    factors_by_condition: dict[int, PairFactor] = {}
    for condition_mask in condition_masks:
        weights = by_condition.get(condition_mask, {})
        total = arithmetic.zero()
        for value in weights.values():
            total = arithmetic.add(total, value)
        counting_weights = (
            tuple(
                (mask, value)
                for mask, value in sorted(weights.items())
                if not arithmetic.is_zero(value)
            )
            if projected_preds
            else ()
        )
        factors_by_condition[condition_mask] = PairFactor(total, counting_weights)

    rows = tuple(
        tuple(
            factors_by_condition[_condition_mask(condition_atoms, left, right)]
            for right in cell_tuple
        )
        for left in cell_tuple
    )
    materialize_ms = (perf_counter() - materialize_started) * 1000
    logger.info(
        "Computed pair factors: backend=%s cnf_vars=%d cells=%d "
        "projected_states=%d condition_states=%d enumerated_models=%d "
        "enumerate_ms=%.3f ganak_ms=%.3f compile_ms=%.3f evaluate_ms=%.3f "
        "materialize_ms=%.3f total_ms=%.3f",
        backend,
        cnf.n_vars,
        len(cell_tuple),
        len(combined),
        len(by_condition),
        enumerated_models,
        enumerate_ms,
        ganak_ms,
        compile_ms,
        evaluate_ms,
        materialize_ms,
        (perf_counter() - started) * 1000,
    )
    return rows


def _enumerate_bounded(
    cnf: TseitinCNF,
    condition_atoms: Sequence[Atom],
    condition_masks: Sequence[int],
    literal_weights: Sequence[WeightPair],
    projection: Mapping[int, int],
    arithmetic: ArithmeticContext,
    *,
    model_limit: int,
) -> tuple[Factor | None, int]:
    """Try exact projected WMC with a globally bounded number of SAT models."""

    original_variables = tuple(cnf.atom_to_var.values())
    condition_variables = tuple(cnf.atom_to_var[atom] for atom in condition_atoms)
    combined: Factor = {}
    model_count = 0
    with Solver(name="cadical195", bootstrap_with=cnf.clauses) as solver:
        for condition_mask in condition_masks:
            assumptions = tuple(
                variable if condition_mask & (1 << index) else -variable
                for index, variable in enumerate(condition_variables)
            )
            while solver.solve(assumptions=assumptions):
                model_count += 1
                if model_count > model_limit:
                    return None, model_count
                positive = {
                    literal for literal in solver.get_model() if literal > 0
                }
                weight = arithmetic.one()
                for variable in original_variables:
                    weight_pair = literal_weights[variable]
                    weight = arithmetic.multiply(
                        weight,
                        weight_pair[0 if variable in positive else 1],
                    )
                mask = 0
                for variable, bit in projection.items():
                    if variable in positive:
                        mask |= 1 << bit
                combined[mask] = arithmetic.add(
                    combined.get(mask, arithmetic.zero()),
                    weight,
                )

                if not original_variables:
                    break
                solver.add_clause(
                    [
                        -variable if variable in positive else variable
                        for variable in original_variables
                    ]
                )
    return combined, model_count


def _condition_atoms(
    cnf: TseitinCNF,
    cell_predicates: Sequence[Predicate],
) -> tuple[Atom, ...]:
    predicate_set = frozenset(cell_predicates)
    return tuple(
        atom
        for atom in cnf.atoms
        if atom.predicate in predicate_set
        and atom.terms
        and (
            all(term == a for term in atom.terms)
            or all(term == b for term in atom.terms)
        )
    )
def _counting_projection(
    cnf: TseitinCNF,
    predicates: Sequence[Predicate],
) -> dict[int, int]:
    result = {}
    for index, predicate in enumerate(predicates):
        for atom, bit in (
            (predicate(b, a), 2 * index),
            (predicate(a, b), 2 * index + 1),
        ):
            variable = cnf.atom_to_var.get(atom)
            if variable is not None:
                result[variable] = bit
    return result


def _free_offdiagonal_factor(
    cnf: TseitinCNF,
    cell_predicates: Sequence[Predicate],
    projected_predicates: Sequence[Predicate],
    predicate_weights: Mapping[Predicate, WeightPair],
    arithmetic: ArithmeticContext,
) -> Factor:
    """Return the exact factor for vocabulary atoms absent from the pair CNF."""

    projected_indices = {
        predicate: index for index, predicate in enumerate(projected_predicates)
    }
    result: Factor = {0: arithmetic.one()}
    one = arithmetic.one()
    for predicate in cell_predicates:
        if predicate.arity != 2:
            continue
        weight_true, weight_false = predicate_weights.get(predicate, (one, one))
        projected_index = projected_indices.get(predicate)
        for atom, projected_bit in (
            (
                predicate(b, a),
                None if projected_index is None else 2 * projected_index,
            ),
            (
                predicate(a, b),
                None if projected_index is None else 2 * projected_index + 1,
            ),
        ):
            if atom in cnf.atom_to_var:
                continue
            if projected_bit is None:
                value = arithmetic.add(weight_true, weight_false)
                choices = {} if arithmetic.is_zero(value) else {0: value}
            else:
                choices = {}
                if not arithmetic.is_zero(weight_false):
                    choices[0] = weight_false
                if not arithmetic.is_zero(weight_true):
                    choices[1 << projected_bit] = weight_true
            result = _factor_multiply(result, choices, arithmetic)
            if not result:
                return {}
    return result


def _literal_weights(
    cnf: TseitinCNF,
    condition_atoms: Sequence[Atom],
    predicate_weights: Mapping[Predicate, WeightPair],
    arithmetic: ArithmeticContext,
) -> tuple[WeightPair, ...]:
    one = arithmetic.one()
    condition_set = frozenset(condition_atoms)
    result = [(one, one) for _ in range(cnf.n_vars + 1)]
    for atom, variable in cnf.atom_to_var.items():
        if atom not in condition_set:
            result[variable] = predicate_weights.get(atom.predicate, (one, one))
    return tuple(result)


def _condition_mask(
    condition_atoms: Sequence[Atom],
    left: Cell,
    right: Cell,
) -> int:
    mask = 0
    for index, atom in enumerate(condition_atoms):
        if all(term == a for term in atom.terms):
            positive = left.is_positive(atom.predicate)
        else:
            positive = right.is_positive(atom.predicate)
        if positive:
            mask |= 1 << index
    return mask


class _SddCircuit:
    def __init__(self, cnf: TseitinCNF, arithmetic: ArithmeticContext):
        self.arithmetic = arithmetic
        self.vtree = Vtree(
            var_count=cnf.n_vars,
            var_order=list(range(1, cnf.n_vars + 1)),
            vtree_type="balanced",
        )
        self.manager = SddManager.from_vtree(self.vtree)
        root = self.manager.true()
        for clause in cnf.clauses:
            clause_node = self.manager.false()
            for literal in clause:
                clause_node = clause_node | self.manager.literal(literal)
            root = root & clause_node
        self.root = root
        self.root.ref()
        self._scopes: dict[int, frozenset[int]] = {}
        self._scope(self.vtree)

    def close(self) -> None:
        self.root.deref()

    def _scope(self, vtree) -> frozenset[int]:
        position = vtree.position()
        cached = self._scopes.get(position)
        if cached is not None:
            return cached
        if vtree.is_leaf():
            result = frozenset((vtree.var(),))
        else:
            result = self._scope(vtree.left()) | self._scope(vtree.right())
        self._scopes[position] = result
        return result

    def evaluate(
        self,
        weights: Sequence[WeightPair],
        projection: Mapping[int, int],
    ) -> Factor:
        memo: dict[tuple[int, int], Factor] = {}

        def free_factor(variables: Iterable[int]) -> Factor:
            result = {0: self.arithmetic.one()}
            for variable in sorted(variables):
                choices = _factor_add(
                    _literal_factor(variable, weights, projection, self.arithmetic),
                    _literal_factor(-variable, weights, projection, self.arithmetic),
                    self.arithmetic,
                )
                result = _factor_multiply(result, choices, self.arithmetic)
            return result

        def visit(node, expected_vtree) -> Factor:
            key = (node.id, expected_vtree.position())
            cached = memo.get(key)
            if cached is not None:
                return cached
            expected_scope = self._scope(expected_vtree)
            if node.is_false():
                result: Factor = {}
            elif node.is_true():
                result = free_factor(expected_scope)
            else:
                actual_vtree = node.vtree()
                actual_scope = self._scope(actual_vtree)
                if actual_scope != expected_scope:
                    if not actual_scope < expected_scope:
                        raise AssertionError("SDD node is outside its expected vtree")
                    result = _factor_multiply(
                        visit(node, actual_vtree),
                        free_factor(expected_scope - actual_scope),
                        self.arithmetic,
                    )
                elif node.is_literal():
                    result = _literal_factor(
                        node.literal,
                        weights,
                        projection,
                        self.arithmetic,
                    )
                else:
                    result = {}
                    for prime, sub in node.elements():
                        element = _factor_multiply(
                            visit(prime, actual_vtree.left()),
                            visit(sub, actual_vtree.right()),
                            self.arithmetic,
                        )
                        result = _factor_add(result, element, self.arithmetic)
            memo[key] = result
            return result

        return visit(self.root, self.vtree)


def _literal_factor(
    literal: int,
    weights: Sequence[WeightPair],
    projection: Mapping[int, int],
    arithmetic: ArithmeticContext,
) -> Factor:
    variable = abs(literal)
    value = weights[variable][0 if literal > 0 else 1]
    if arithmetic.is_zero(value):
        return {}
    mask = 1 << projection[variable] if literal > 0 and variable in projection else 0
    return {mask: value}


def _factor_add(
    left: Factor,
    right: Factor,
    arithmetic: ArithmeticContext,
) -> Factor:
    result = dict(left)
    for mask, value in right.items():
        result[mask] = arithmetic.add(
            result.get(mask, arithmetic.zero()),
            value,
        )
        if arithmetic.is_zero(result[mask]):
            del result[mask]
    return result


def _factor_multiply(
    left: Factor,
    right: Factor,
    arithmetic: ArithmeticContext,
) -> Factor:
    result: Factor = {}
    for left_mask, left_value in left.items():
        for right_mask, right_value in right.items():
            if left_mask & right_mask:
                raise AssertionError("Decomposable SDD repeated a projection bit")
            mask = left_mask | right_mask
            result[mask] = arithmetic.add(
                result.get(mask, arithmetic.zero()),
                arithmetic.multiply(left_value, right_value),
            )
    return result


__all__ = ["compute_pair_factors"]
