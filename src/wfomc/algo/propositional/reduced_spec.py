"""Reduction-first propositional algorithm spec."""

from __future__ import annotations

from wfomc.algo.core import (
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    PreparedBranch,
    option_resolver,
)
from wfomc.engine.compilation import (
    CompiledReducedProblem,
    compile_reduced_problem,
    instantiate_reduced_problem,
)
from wfomc.engine.features import FeatureSet
from wfomc.fol.grounding import LinearOrderEncoding
from wfomc.problem import Domain, Problem
from wfomc.reduction import reduce_problem

from .reduced_input import (
    ReducedPropositionalInputTemplate,
    build_input_template,
    instantiate_input_template,
)
from .solve import solve


def _choose_unary_evidence(
    features: FeatureSet,
    linear_order_encoding: LinearOrderEncoding,
) -> EvidenceStrategy:
    if not features.has_unary_evidence:
        return EvidenceStrategy.NONE
    if features.has_linear_order and linear_order_encoding is LinearOrderEncoding.PIN:
        return EvidenceStrategy.CCS
    return EvidenceStrategy.GROUND_UNITS


def compile_branches(
    problem: Problem,
    options: AlgoOptions,
) -> tuple[object, ...]:
    return tuple(
        compile_reduced_problem(reduced, options)
        for reduced in reduce_problem(problem, options)
    )


def branch_applies(branch: object, domain: Domain) -> bool:
    if not isinstance(branch, CompiledReducedProblem):
        raise TypeError("Reduced propositional branch has an invalid type")
    return branch.reduced_problem.applies(domain)


def build_propositional_input_template(
    branch: object,
    input_variant: object,
    _options: AlgoOptions,
) -> ReducedPropositionalInputTemplate:
    if not isinstance(branch, CompiledReducedProblem):
        raise TypeError("Reduced propositional branch has an invalid type")
    if input_variant is not None:
        raise TypeError("Reduced propositional input has no structural variants")
    return build_input_template(branch)


def instantiate_branch(
    branch: object,
    input_template: object,
    domain: Domain,
    options: AlgoOptions,
) -> PreparedBranch | None:
    if not isinstance(branch, CompiledReducedProblem):
        raise TypeError("Reduced propositional branch has an invalid type")
    if not isinstance(input_template, ReducedPropositionalInputTemplate):
        raise TypeError("Reduced propositional input template has an invalid type")
    instantiated = instantiate_reduced_problem(branch, domain)
    if instantiated is None:
        return None
    concrete, decoder = instantiated
    return PreparedBranch(
        branch.reduced_problem,
        instantiate_input_template(
            input_template,
            concrete,
            options=options,
        ),
        decoder,
    )


SPEC = AlgoSpec(
    name=AlgoName.PROPOSITIONAL_REDUCED,
    resolve_options=option_resolver(
        algo=AlgoName.PROPOSITIONAL_REDUCED,
        default_unary_evidence=_choose_unary_evidence,
        supported_unary_evidence=(
            EvidenceStrategy.GROUND_UNITS,
            EvidenceStrategy.CCS,
        ),
        supports_linear_order=True,
        supports_predk_or_circular=True,
    ),
    solve=solve,
    compile_branches=compile_branches,
    branch_applies=branch_applies,
    build_input_template=build_propositional_input_template,
    instantiate_branch=instantiate_branch,
    maturity=AlgoMaturity.BETA,
    external_requirements=("Ganak executable",),
)


__all__ = ["SPEC"]
