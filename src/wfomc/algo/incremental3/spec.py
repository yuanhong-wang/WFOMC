"""Incremental3 algorithm spec."""

from __future__ import annotations

from wfomc.algo.core import (
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    ExistentialStrategy,
    PreparedBranch,
    compile_reduced_problem,
    reduce_unary_evidence_for_options,
    option_resolver,
)
from .input import build_input
from .solve import solve
from wfomc.problem import Problem
from wfomc.reduction import (
    reduce_cardinality_constraints,
    reduce_existential_quantifiers,
    apply_reductions,
)


REDUCTIONS = (
    reduce_unary_evidence_for_options,
    reduce_existential_quantifiers,
    reduce_cardinality_constraints,
)


def prepare(problem: Problem, options: AlgoOptions) -> tuple[PreparedBranch, ...]:
    from wfomc.algo.incremental3.counting_state import (
        build_counting_state_for_normal_form,
    )

    reduced = apply_reductions(problem, REDUCTIONS, options)
    prepared = []
    for branch in reduced.problems:
        counting_state, unary_masks = build_counting_state_for_normal_form(
            branch.problem.normal_form,
            domain_size=len(branch.problem.domain),
        )
        compiled, features = compile_reduced_problem(branch.problem, options)
        algo_input = build_input(
            compiled,
            options=options,
            features=features,
            counting_state=counting_state,
            unary_cardinality_masks=unary_masks,
        )
        prepared.append(PreparedBranch(branch.problem, algo_input, branch.decoder))
    return tuple(prepared)


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
    prepare=prepare,
    solve=solve,
)


__all__ = ["SPEC"]
