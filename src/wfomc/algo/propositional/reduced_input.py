"""Reduction-first ground-CNF input builder for the propositional solver."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace
from typing import TYPE_CHECKING

from wfomc.algo.core import (
    AlgoOptions,
    EvidenceStrategy,
    ReducedInputTemplate,
)
from wfomc.stages import (
    CompiledBranchInstance,
    CompiledReducedBranch,
    FeatureSet,
)

from .input import GroundCNFInput

if TYPE_CHECKING:
    from wfomc.fol.syntax import Literal


@dataclass(frozen=True)
class ReducedPropositionalInputTemplate(ReducedInputTemplate):
    """Reusable reduced branch metadata for per-domain grounding."""

    compiled: CompiledReducedBranch
    options: AlgoOptions

    def instantiate(
        self,
        concrete: CompiledBranchInstance,
    ) -> GroundCNFInput:
        """Ground one reduced branch for a concrete domain."""

        return build_reduced_input(
            concrete,
            options=self.options,
            features=self.compiled.feature_set,
        )


def build_input_template(
    compiled: CompiledReducedBranch,
    options: AlgoOptions,
) -> ReducedPropositionalInputTemplate:
    """Wrap a reduced numeric branch for staged grounding."""

    return ReducedPropositionalInputTemplate(compiled, options)


def build_reduced_input(
    reduced: CompiledBranchInstance,
    *,
    options: AlgoOptions,
    features: FeatureSet,
) -> GroundCNFInput:
    """Ground one fully reduced quantifier-free problem."""

    from wfomc.fol.grounding import (
        ground_qf_formula,
        linear_order_clauses,
        resolve_linear_order_encoding,
    )

    if reduced.sentence is None:
        raise RuntimeError("reduced problem has no quantifier-free formula")
    encoding = resolve_linear_order_encoding(options.linear_order_encoding)
    domain = tuple(sorted(reduced.domain, key=lambda constant: constant.name))
    atom_to_id, id_to_predicate, clauses, unsat = ground_qf_formula(
        reduced.sentence,
        list(domain),
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
    if evidence and options.evidence_strategy is EvidenceStrategy.GROUND_UNITS:
        evidence_predicates = {item.atom.predicate for item in evidence}
        for predicate in sorted(
            evidence_predicates,
            key=lambda item: (item.name, item.arity),
        ):
            for arguments in product(domain, repeat=predicate.arity):
                atom = predicate(*arguments)
                if atom not in atom_to_id:
                    variable = fresh()
                    atom_to_id[atom] = variable
                    id_to_predicate[variable] = predicate
        for item in sorted(evidence, key=_literal_sort_key):
            variable = atom_to_id[item.atom]
            evidence_unit_clauses.append(
                frozenset((variable if item.positive else -variable,))
            )

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
        variable: weight_map.get(predicate, (one, one))
        for variable, predicate in id_to_predicate.items()
    }
    cnf = tuple(clauses) + tuple(evidence_unit_clauses) + tuple(order_clauses)
    return GroundCNFInput(
        arithmetic=reduced.arithmetic,
        cnf=cnf,
        literal_weights=literal_weights,
        linear_order_encoding=encoding,
        leq_present=features.leq_predicate is not None,
    )


def _literal_sort_key(literal: "Literal") -> tuple[object, ...]:
    return (
        literal.atom.predicate.name,
        tuple(str(term) for term in literal.atom.terms),
        literal.positive,
    )


__all__ = [
    "ReducedPropositionalInputTemplate",
    "build_input_template",
    "build_reduced_input",
]
