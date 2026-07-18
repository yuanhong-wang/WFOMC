"""Incremental3 algorithm spec."""

from __future__ import annotations

from wfomc.algo.core import (
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    ExistentialStrategy,
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
    Incremental3InputTemplate,
    Incremental3InputVariant,
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
        for reduced in reduce_problem(
            problem,
            options,
            reduce_counting_quantifiers=False,
        )
    )


def branch_applies(branch: object, domain: Domain) -> bool:
    if not isinstance(branch, CompiledReducedProblem):
        raise TypeError("Incremental3 compiled branch has an invalid type")
    return branch.reduced_problem.applies(domain)


def input_template_variant(
    branch: object,
    domain: Domain,
) -> Incremental3InputVariant:
    if not isinstance(branch, CompiledReducedProblem):
        raise TypeError("Incremental3 compiled branch has an invalid type")
    from .counting_state import build_counting_state_for_normal_form

    counting_state, _unary_masks = build_counting_state_for_normal_form(
        branch.reduced_problem.normal_form,
        domain_size=domain.size,
    )
    return Incremental3InputVariant(
        evidence=select_input_variant(branch, domain),
        counting_state=counting_state,
    )


def build_incremental3_input_template(
    branch: object,
    input_variant: object,
    options: AlgoOptions,
) -> Incremental3InputTemplate:
    if not isinstance(branch, CompiledReducedProblem):
        raise TypeError("Incremental3 compiled branch has an invalid type")
    if not isinstance(input_variant, Incremental3InputVariant):
        raise TypeError("Incremental3 input variant has an invalid type")
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
        raise TypeError("Incremental3 compiled branch has an invalid type")
    if not isinstance(input_template, Incremental3InputTemplate):
        raise TypeError("Incremental3 input template has an invalid type")
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
    name=AlgoName.INCREMENTAL3,
    resolve_options=option_resolver(
        algo=AlgoName.INCREMENTAL3,
        default_unary_evidence=EvidenceStrategy.LIFTED_PROFILES,
        supported_unary_evidence=(
            EvidenceStrategy.CCS,
            EvidenceStrategy.LIFTED_PROFILES,
        ),
        default_existential_strategy=ExistentialStrategy.COUNTING,
        supported_existential_strategies=(
            ExistentialStrategy.COUNTING,
            ExistentialStrategy.SKOLEM,
        ),
        supports_linear_order=True,
        supports_mod_counting=True,
    ),
    solve=solve,
    compile_branches=compile_branches,
    branch_applies=branch_applies,
    input_template_variant=input_template_variant,
    build_input_template=build_incremental3_input_template,
    instantiate_branch=instantiate_branch,
)


__all__ = ["SPEC"]
