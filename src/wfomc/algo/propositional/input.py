"""Direct source-grounding input contract owned by the propositional algorithm."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from types import SimpleNamespace
from typing import TYPE_CHECKING

from wfomc.algo.core import (
    AlgoInput,
    AlgoName,
    AlgoOptions,
)
from wfomc.arithmetic import ArithmeticValue
from wfomc.cardinality_constraints import Comparator
from wfomc.engine.compilation import CompiledSourceProblem
from wfomc.engine.features import FeatureSet
from wfomc.errors import UnsupportedFeatureError
from wfomc.problem import Domain, Problem

if TYPE_CHECKING:
    from wfomc.fol.grounding import LinearOrderEncoding
    from wfomc.fol.syntax import Atom, Constant, Predicate


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundCNFInput(AlgoInput):
    cnf: tuple[frozenset[int], ...] = ()
    literal_weights: dict[int, tuple[ArithmeticValue, ArithmeticValue]] = field(
        default_factory=dict
    )
    evidence_unit_clauses: tuple[frozenset[int], ...] = ()
    cardinality_clauses: tuple[frozenset[int], ...] = ()
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


@dataclass(frozen=True)
class PropositionalInputTemplate:
    """Compiled source formula and weights reused across ground domains."""

    compiled: CompiledSourceProblem


def build_input_template(
    compiled: CompiledSourceProblem,
) -> PropositionalInputTemplate:
    """Wrap one domain-free source compilation for staged grounding."""

    return PropositionalInputTemplate(compiled)


def instantiate_input_template(
    template: PropositionalInputTemplate,
    domain: Domain,
    *,
    options: AlgoOptions,
) -> GroundCNFInput:
    """Ground a compiled source problem for one concrete domain."""

    return _ground_input(
        template.compiled,
        domain,
        options=options,
        features=template.compiled.feature_set,
    )


def _ground_input(
    compiled: CompiledSourceProblem,
    domain_instance: Domain,
    *,
    options: AlgoOptions,
    features: FeatureSet,
) -> GroundCNFInput:
    """Ground one compiled source problem into model-preserving CNF."""

    from flint import fmpq_mpoly, fmpq_poly
    from wfomc.fol.analysis import predicates
    from wfomc.fol.cnf import encode_tseitin
    from wfomc.fol.grounding import (
        at_least_k_clauses,
        at_most_k_clauses,
        exactly_k_clauses,
        ground_source_formula,
        linear_order_clauses,
        resolve_linear_order_encoding,
    )

    problem = compiled.problem
    arithmetic = compiled.arithmetic
    compiled_weights = compiled.weights
    encoding = resolve_linear_order_encoding(options.linear_order_encoding)
    domain = tuple(sorted(domain_instance.elements, key=str))
    grounded = ground_source_formula(problem.sentence, domain)
    tseitin = encode_tseitin(grounded)

    atom_to_id = dict(tseitin.atom_to_var)
    id_to_predicate = {
        variable: atom.predicate for atom, variable in atom_to_id.items()
    }
    formula_clauses = [frozenset(clause) for clause in tseitin.clauses]
    next_id = tseitin.n_vars

    def fresh() -> int:
        nonlocal next_id
        next_id += 1
        return next_id

    def ensure_atom(atom: Atom) -> int:
        positive = atom.make_positive()
        variable = atom_to_id.get(positive)
        if variable is None:
            variable = fresh()
            atom_to_id[positive] = variable
            id_to_predicate[variable] = positive.predicate
        return variable

    predicate_universe = set(predicates(problem.sentence)) | set(problem.weights)
    for evidence_item in (
        *problem.evidence.unary.literals,
        *problem.evidence.binary.literals,
    ):
        predicate_universe.add(evidence_item.predicate)
    for constraint in problem.cardinality_constraints.constraints:
        predicate_universe.update(term.predicate for term in constraint.terms)
    _ground_predicate_universe(predicate_universe, domain, ensure_atom)

    evidence_unit_clauses = _ground_evidence(problem, domain, ensure_atom)
    cardinality_clauses = _ground_cardinality_constraints(
        problem,
        domain,
        ensure_atom,
        at_most_k_clauses,
        exactly_k_clauses,
        at_least_k_clauses,
    )

    order_view = SimpleNamespace(
        formula=problem.sentence,
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

    one = arithmetic.one()
    literal_weights = {
        variable: compiled_weights.get(predicate, (one, one))
        for variable, predicate in id_to_predicate.items()
    }
    symbolic = any(
        isinstance(weight, (fmpq_poly, fmpq_mpoly))
        for pair in literal_weights.values()
        for weight in pair
    )
    cnf = (
        tuple(formula_clauses)
        + tuple(cardinality_clauses)
        + tuple(evidence_unit_clauses)
        + tuple(order_clauses)
    )
    logger.info(
        "Direct propositional grounding: domain=%d atoms=%d formula_clauses=%d "
        "cardinality_clauses=%d evidence_clauses=%d order_clauses=%d",
        len(domain),
        len(atom_to_id),
        len(formula_clauses),
        len(cardinality_clauses),
        len(evidence_unit_clauses),
        len(order_clauses),
    )
    return GroundCNFInput(
        algo=AlgoName.PROPOSITIONAL,
        options=options,
        arithmetic=arithmetic,
        cnf=cnf,
        literal_weights=literal_weights,
        evidence_unit_clauses=tuple(evidence_unit_clauses),
        cardinality_clauses=tuple(cardinality_clauses),
        domain_size=len(domain),
        domain=domain,
        atom_to_id=atom_to_id,
        id_to_predicate=id_to_predicate,
        clauses=tuple(formula_clauses),
        order_unit_clauses=tuple(order_clauses),
        linear_order_encoding=encoding,
        symbolic=symbolic,
        leq_present=features.leq_predicate is not None,
    )


def _ground_evidence(
    problem: Problem,
    domain: tuple[Constant, ...],
    ensure_atom,
) -> list[frozenset[int]]:
    literals = [
        *(item.to_ground_literal() for item in problem.evidence.unary.literals),
        *(item.to_ground_literal() for item in problem.evidence.binary.literals),
    ]
    domain_set = set(domain)
    for literal in literals:
        outside = [term for term in literal.atom.terms if term not in domain_set]
        if outside:
            raise ValueError(
                "Evidence constants must belong to the problem domain: "
                + ", ".join(map(str, outside))
            )
    predicates = {literal.atom.predicate for literal in literals}
    for predicate in sorted(predicates, key=lambda item: (item.name, item.arity)):
        for arguments in product(domain, repeat=predicate.arity):
            ensure_atom(predicate(*arguments))
    clauses = []
    for literal in sorted(
        literals,
        key=lambda item: (
            str(item.atom.predicate),
            tuple(str(term) for term in item.atom.terms),
            item.positive,
        ),
    ):
        variable = ensure_atom(literal.atom)
        clauses.append(frozenset((variable if literal.positive else -variable,)))
    return clauses


def _ground_predicate_universe(
    predicates: set[object],
    domain: tuple[Constant, ...],
    ensure_atom,
) -> None:
    """Allocate every relation entry belonging to the source vocabulary."""

    for predicate in sorted(
        predicates,
        key=lambda item: (str(item), getattr(item, "arity", -1)),
    ):
        arity = getattr(predicate, "arity", None)
        if not isinstance(arity, int):
            continue
        for arguments in product(domain, repeat=arity):
            ensure_atom(predicate(*arguments))


def _ground_cardinality_constraints(
    problem: Problem,
    domain: tuple[Constant, ...],
    ensure_atom,
    at_most,
    exactly,
    at_least,
) -> list[frozenset[int]]:
    clauses: list[frozenset[int]] = []
    builders = {
        Comparator.LE: at_most,
        Comparator.EQ: exactly,
        Comparator.GE: at_least,
    }
    for constraint in problem.cardinality_constraints.constraints:
        coefficients: defaultdict[object, int] = defaultdict(int)
        for term in constraint.terms:
            coefficients[term.predicate] += term.coefficient
        coefficients = defaultdict(
            int,
            {
                predicate: coefficient
                for predicate, coefficient in coefficients.items()
                if coefficient
            },
        )
        if (
            len(coefficients) != 1
            or next(iter(coefficients.values()), None) != 1
            or constraint.comparator not in builders
        ):
            raise UnsupportedFeatureError(
                "direct propositional grounding currently supports only "
                "|P| <= k, |P| = k, and |P| >= k global cardinality constraints"
            )
        predicate = next(iter(coefficients))
        arity = getattr(predicate, "arity", None)
        if not isinstance(arity, int):
            raise UnsupportedFeatureError(
                "direct propositional cardinality grounding requires a typed predicate"
            )
        literals = tuple(
            ensure_atom(predicate(*arguments))
            for arguments in product(domain, repeat=arity)
        )
        clauses.extend(builders[constraint.comparator](literals, constraint.rhs))
    return clauses


__all__ = [
    "GroundCNFInput",
    "PropositionalInputTemplate",
    "build_input_template",
    "instantiate_input_template",
]
