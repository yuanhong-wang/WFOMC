"""Direct-grounding propositional algorithm spec."""

from __future__ import annotations

from dataclasses import replace

from wfomc.algo.core import (
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    ExistentialStrategy,
    PreparedBranch,
    option_resolver,
)
from wfomc.engine.features import FeatureSet, analyze_features
from wfomc.errors import UnsupportedFeatureError
from wfomc.fol.grounding import LinearOrderEncoding
from wfomc.problem import Problem
from wfomc.reduction import identity_decoder

from .input import build_input
from .solve import solve


_BASE_RESOLVE_OPTIONS = option_resolver(
    algo=AlgoName.PROPOSITIONAL,
    default_unary_evidence=EvidenceStrategy.GROUND_UNITS,
    supported_unary_evidence=(EvidenceStrategy.GROUND_UNITS,),
    default_existential_strategy=ExistentialStrategy.GROUND,
    supported_existential_strategies=(ExistentialStrategy.GROUND,),
    supports_linear_order=True,
    supports_predk_or_circular=True,
    supports_mod_counting=True,
    supports_binary_evidence=True,
)


def resolve_options(
    features: FeatureSet,
    options: AlgoOptions | None = None,
) -> AlgoOptions:
    """Choose direct-grounding options without symmetry-breaking evidence loss."""

    requested = options if options is not None else AlgoOptions()
    asymmetric = (
        features.has_named_constants
        or features.has_unary_evidence
        or features.has_binary_evidence
    )
    if (
        asymmetric
        and features.has_linear_order
        and requested.linear_order_encoding is None
    ):
        requested = replace(
            requested,
            linear_order_encoding=LinearOrderEncoding.AXIOMS,
        )
    resolved = _BASE_RESOLVE_OPTIONS(features, requested)
    if (
        asymmetric
        and features.has_linear_order
        and resolved.linear_order_encoding is LinearOrderEncoding.PIN
    ):
        raise UnsupportedFeatureError(
            "propositional pin-and-multiply is not sound with named constants "
            "or ground evidence; use linear_order_encoding='axioms'"
        )
    return resolved


def prepare(problem: Problem, options: AlgoOptions) -> tuple[PreparedBranch, ...]:
    features = analyze_features(problem)
    if (
        options.linear_order_encoding is LinearOrderEncoding.AXIOMS
        and any(order > 1 for order, _predicate in features.predecessor_predicates)
    ):
        raise UnsupportedFeatureError(
            "direct propositional order axioms currently support only PRED/PRED1"
        )
    algo_input = build_input(problem, options=options, features=features)
    return (PreparedBranch(problem, algo_input, identity_decoder),)


SPEC = AlgoSpec(
    name=AlgoName.PROPOSITIONAL,
    resolve_options=resolve_options,
    prepare=prepare,
    solve=solve,
    maturity=AlgoMaturity.BETA,
    external_requirements=("Ganak executable",),
)


__all__ = ["SPEC"]
