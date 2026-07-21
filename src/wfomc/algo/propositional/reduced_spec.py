"""Reduction-first propositional algorithm spec."""

from __future__ import annotations

from collections.abc import Hashable

from wfomc.algo.core import (
    AlgoBranch,
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    option_resolver,
)
from wfomc.fol.grounding import LinearOrderEncoding
from wfomc.stages import CompiledReducedBranch, FeatureSet

from .reduced_input import (
    ReducedPropositionalInputTemplate,
    build_input_template,
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


def build_propositional_input_template(
    branch: AlgoBranch,
    input_key: Hashable,
    options: AlgoOptions,
) -> ReducedPropositionalInputTemplate:
    if not isinstance(branch, CompiledReducedBranch):
        raise TypeError("Reduced propositional requires a compiled reduced branch")
    if input_key is not None:
        raise TypeError("Reduced propositional input has no structural keys")
    return build_input_template(branch, options)


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
    build_input_template=build_propositional_input_template,
    maturity=AlgoMaturity.BETA,
)


__all__ = ["SPEC"]
