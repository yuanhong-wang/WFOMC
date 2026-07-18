"""Incremental algorithm spec."""

from __future__ import annotations

from wfomc.algo.core import (
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    PreparedBranch,
    option_resolver,
)
from wfomc.cell_graph.staging import select_input_variant
from wfomc.engine.compilation import (
    CompiledReducedProblem,
    compile_reduced_problem,
    instantiate_reduced_problem,
)
from .input import (
    IncrementalInputTemplate,
    build_input_template,
    instantiate_input_template,
)
from .solve import solve
from wfomc.problem import Domain, Problem
from wfomc.reduction import reduce_problem


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
        raise TypeError("Incremental compiled branch has an invalid type")
    return branch.reduced_problem.applies(domain)


def input_template_variant(branch: object, domain: Domain) -> object:
    if not isinstance(branch, CompiledReducedProblem):
        raise TypeError("Incremental compiled branch has an invalid type")
    return select_input_variant(branch, domain)


def build_incremental_input_template(
    branch: object,
    input_variant: object,
    options: AlgoOptions,
) -> IncrementalInputTemplate:
    if not isinstance(branch, CompiledReducedProblem):
        raise TypeError("Incremental compiled branch has an invalid type")
    return build_input_template(
        branch,
        input_variant=input_variant,
        options=options,
    )


def instantiate_branch(
    branch: object,
    input_template: object,
    domain: Domain,
    options: AlgoOptions,
) -> PreparedBranch | None:
    if not isinstance(branch, CompiledReducedProblem):
        raise TypeError("Incremental compiled branch has an invalid type")
    if not isinstance(input_template, IncrementalInputTemplate):
        raise TypeError("Incremental input template has an invalid type")
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
    name=AlgoName.INCREMENTAL,
    resolve_options=option_resolver(
        algo=AlgoName.INCREMENTAL,
        default_unary_evidence=EvidenceStrategy.LIFTED_PROFILES,
        supported_unary_evidence=(
            EvidenceStrategy.CCS,
            EvidenceStrategy.LIFTED_PROFILES,
        ),
        supports_linear_order=True,
        supports_predk_or_circular=True,
    ),
    solve=solve,
    compile_branches=compile_branches,
    branch_applies=branch_applies,
    input_template_variant=input_template_variant,
    build_input_template=build_incremental_input_template,
    instantiate_branch=instantiate_branch,
)


__all__ = ["SPEC"]
