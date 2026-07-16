"""Reduction-first propositional algorithm spec."""

from __future__ import annotations

from wfomc.algo.core import (
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    PreparedBranch,
    compile_reduced_problem,
    option_resolver,
    reduce_unary_evidence_for_options,
)
from wfomc.engine.features import FeatureSet
from wfomc.fol.grounding import LinearOrderEncoding
from wfomc.problem import Problem
from wfomc.reduction import (
    apply_reductions,
    reduce_cardinality_constraints,
    reduce_counting_quantifiers,
    reduce_existential_quantifiers,
)

from .reduced_input import build_reduced_input
from .solve import solve


REDUCTIONS = (
    reduce_unary_evidence_for_options,
    reduce_counting_quantifiers,
    reduce_existential_quantifiers,
    reduce_cardinality_constraints,
)


def _choose_unary_evidence(
    features: FeatureSet,
    linear_order_encoding: LinearOrderEncoding,
) -> EvidenceStrategy:
    if not features.has_unary_evidence:
        return EvidenceStrategy.NONE
    if (
        features.has_linear_order
        and linear_order_encoding is LinearOrderEncoding.PIN
    ):
        return EvidenceStrategy.CCS
    return EvidenceStrategy.GROUND_UNITS


def prepare(problem: Problem, options: AlgoOptions) -> tuple[PreparedBranch, ...]:
    reduced = apply_reductions(problem, REDUCTIONS, options)
    prepared = []
    for branch in reduced.problems:
        compiled, features = compile_reduced_problem(branch.problem, options)
        algo_input = build_reduced_input(
            compiled,
            options=options,
            features=features,
        )
        prepared.append(PreparedBranch(branch.problem, algo_input, branch.decoder))
    return tuple(prepared)


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
    prepare=prepare,
    solve=solve,
    maturity=AlgoMaturity.BETA,
    external_requirements=("Ganak executable",),
)


__all__ = ["SPEC"]
