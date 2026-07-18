"""Direct-grounding propositional algorithm spec."""

from __future__ import annotations

from dataclasses import replace

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
    CompiledSourceProblem,
    compile_source_problem,
)
from wfomc.engine.features import FeatureSet
from wfomc.errors import UnsupportedFeatureError
from wfomc.fol.grounding import LinearOrderEncoding
from wfomc.problem import Domain, Problem

from .input import (
    PropositionalInputTemplate,
    build_input_template,
    instantiate_input_template,
)
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


def _identity_decoder(result: object, **_: object) -> object:
    return result


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


def compile_branches(
    problem: Problem,
    options: AlgoOptions,
) -> tuple[object, ...]:
    compiled = compile_source_problem(problem, options)
    if options.linear_order_encoding is LinearOrderEncoding.AXIOMS and any(
        order > 1 for order, _predicate in compiled.feature_set.predecessor_predicates
    ):
        raise UnsupportedFeatureError(
            "direct propositional order axioms currently support only PRED/PRED1"
        )
    return (compiled,)


def build_propositional_input_template(
    branch: object,
    input_variant: object,
    _options: AlgoOptions,
) -> PropositionalInputTemplate:
    if not isinstance(branch, CompiledSourceProblem):
        raise TypeError("Propositional compiled branch has an invalid type")
    if input_variant is not None:
        raise TypeError("Propositional input does not use structural variants")
    return build_input_template(branch)


def instantiate_branch(
    branch: object,
    input_template: object,
    domain: Domain,
    options: AlgoOptions,
) -> PreparedBranch:
    if not isinstance(branch, CompiledSourceProblem):
        raise TypeError("Propositional compiled branch has an invalid type")
    if not isinstance(input_template, PropositionalInputTemplate):
        raise TypeError("Propositional input template has an invalid type")
    algo_input = instantiate_input_template(
        input_template,
        domain,
        options=options,
    )
    return PreparedBranch(branch.problem, algo_input, _identity_decoder)


SPEC = AlgoSpec(
    name=AlgoName.PROPOSITIONAL,
    resolve_options=resolve_options,
    solve=solve,
    compile_branches=compile_branches,
    build_input_template=build_propositional_input_template,
    instantiate_branch=instantiate_branch,
    maturity=AlgoMaturity.BETA,
    external_requirements=("Ganak executable",),
)


__all__ = ["SPEC"]
