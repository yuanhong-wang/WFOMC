"""Input contract owned by the propositional algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from types import SimpleNamespace
from typing import TYPE_CHECKING

from wfomc.algo.core import AlgoInput, AlgoOptions, EvidenceStrategy
from wfomc.arithmetic import ArithmeticValue
from wfomc.problem import CompiledProblem
from wfomc.engine.features import FeatureSet

if TYPE_CHECKING:
    from wfomc.fol.grounding import LinearOrderEncoding
    from wfomc.fol.syntax import Atom, Constant, Predicate


@dataclass(frozen=True)
class GroundCNFInput(AlgoInput):
    cnf: tuple[frozenset[int], ...] = ()
    literal_weights: dict[int, tuple[ArithmeticValue, ArithmeticValue]] = field(
        default_factory=dict
    )
    evidence_unit_clauses: tuple[frozenset[int], ...] = ()
    domain_size: int = 0
    domain: tuple[Constant, ...] = ()
    atom_to_id: dict[Atom, int] = field(default_factory=dict)
    id_to_predicate: dict[int, Predicate] = field(default_factory=dict)
    clauses: tuple[frozenset[int], ...] = ()
    order_unit_clauses: tuple[frozenset[int], ...] = ()
    linear_order_encoding: LinearOrderEncoding | None = None
    symbolic: bool = False
    leq_present: bool = False

    def include_order_factorial(self) -> bool:
        from wfomc.fol.grounding import LinearOrderEncoding

        return not (
            self.linear_order_encoding is LinearOrderEncoding.AXIOMS
            and self.leq_present
        )


def build_input(
    reduced: CompiledProblem,
    *,
    options: AlgoOptions,
    features: FeatureSet,
) -> GroundCNFInput:
    from flint import fmpq_mpoly, fmpq_poly
    from wfomc.fol.grounding import (
        ground_qf_formula,
        linear_order_clauses,
        resolve_linear_order_encoding,
    )

    if reduced.sentence is None:
        raise RuntimeError("reduced problem has no quantifier-free formula")
    encoding = resolve_linear_order_encoding(options.linear_order_encoding)
    domain = tuple(sorted(reduced.domain, key=lambda const: const.name))
    atom_to_id, id_to_predicate, clauses, unsat = ground_qf_formula(
        reduced.sentence, list(domain)
    )
    if unsat:
        clauses = [frozenset()]
    evidence_unit_clauses: list[frozenset[int]] = []
    next_id = len(atom_to_id)

    def fresh() -> int:
        nonlocal next_id
        next_id += 1
        return next_id

    evidence = tuple(
        item.to_ground_literal() for item in reduced.evidence.unary.literals
    )
    if evidence and options.evidence_strategy == EvidenceStrategy.GROUND_UNITS:
        evidence_preds = {_literal_predicate(item) for item in evidence}
        for predicate in sorted(
            evidence_preds, key=lambda item: (item.name, item.arity)
        ):
            for args in product(domain, repeat=predicate.arity):
                atom = predicate(*args)
                if atom not in atom_to_id:
                    vid = fresh()
                    atom_to_id[atom] = vid
                    id_to_predicate[vid] = predicate
        for item in sorted(
            evidence,
            key=lambda value: (
                _literal_predicate(value).name,
                str(_literal_terms(value)),
                value.positive,
            ),
        ):
            vid = atom_to_id[item.atom]
            evidence_unit_clauses.append(frozenset((vid if item.positive else -vid,)))

    order_view = SimpleNamespace(
        formula=reduced.sentence,
        leq_pred=features.leq_predicate,
        predecessor_preds=dict(features.predecessor_predicates),
        circular_predecessor_pred=features.circular_predecessor_predicate,
    )
    order_clauses = linear_order_clauses(
        order_view,
        list(domain),
        atom_to_id,
        id_to_predicate,
        fresh,
        encoding,
    )
    one = reduced.arithmetic.one()
    weight_map = dict(reduced.weights)
    literal_weights = {
        vid: weight_map.get(predicate, (one, one))
        for vid, predicate in id_to_predicate.items()
    }
    symbolic = any(
        isinstance(weight, (fmpq_poly, fmpq_mpoly))
        for pair in literal_weights.values()
        for weight in pair
    )
    cnf = tuple(clauses) + tuple(evidence_unit_clauses) + tuple(order_clauses)
    return GroundCNFInput(
        algo=None,
        options=options,
        arithmetic=reduced.arithmetic,
        cnf=cnf,
        literal_weights=literal_weights,
        evidence_unit_clauses=tuple(evidence_unit_clauses),
        domain_size=len(domain),
        domain=domain,
        atom_to_id=dict(atom_to_id),
        id_to_predicate=dict(id_to_predicate),
        clauses=tuple(clauses),
        order_unit_clauses=tuple(order_clauses),
        linear_order_encoding=encoding,
        symbolic=symbolic,
        leq_present=features.leq_predicate is not None,
    )


def _literal_predicate(literal: object) -> object:
    return literal.atom.predicate


def _literal_terms(literal: object) -> tuple[object, ...]:
    return tuple(literal.atom.terms)


__all__ = ["GroundCNFInput", "build_input"]
