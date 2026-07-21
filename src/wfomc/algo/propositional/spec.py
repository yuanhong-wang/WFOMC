"""Direct-grounding propositional algorithm spec."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import replace

from wfomc.algo.core import (
    AlgoBranch,
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    EvidenceStrategy,
    option_resolver,
)
from wfomc.errors import UnsupportedFeatureError
from wfomc.fol.grounding import LinearOrderEncoding
from wfomc.stages import FeatureSet, GroundingProblem

from .input import PropositionalInputTemplate, build_input_template
from .solve import solve


_BASE_RESOLVE_OPTIONS = option_resolver(
    algo=AlgoName.PROPOSITIONAL,
    default_unary_evidence=EvidenceStrategy.GROUND_UNITS,
    supported_unary_evidence=(EvidenceStrategy.GROUND_UNITS,),
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
    if (
        resolved.linear_order_encoding is LinearOrderEncoding.AXIOMS
        and any(order > 1 for order, _predicate in features.predecessor_predicates)
    ):
        raise UnsupportedFeatureError(
            "direct propositional order axioms currently support only PRED/PRED1"
        )
    return resolved


def build_propositional_input_template(
    grounding_problem: AlgoBranch,
    input_key: Hashable,
    options: AlgoOptions,
) -> PropositionalInputTemplate:
    if not isinstance(grounding_problem, GroundingProblem):
        raise TypeError("Propositional requires a grounding problem")
    if input_key is not None:
        raise TypeError("Propositional input does not use structural keys")
    return build_input_template(grounding_problem, options)


SPEC = AlgoSpec(
    name=AlgoName.PROPOSITIONAL,
    resolve_options=resolve_options,
    solve=solve,
    build_input_template=build_propositional_input_template,
    uses_reduction=False,
    maturity=AlgoMaturity.BETA,
)


__all__ = ["SPEC"]
