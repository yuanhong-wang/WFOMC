"""Domain-free numeric compilation and concrete branch instantiation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wfomc.arithmetic import (
    ArithmeticContext,
    choose_arithmetic_backend,
)
from wfomc.errors import ArithmeticBackendError
from wfomc.weights import (
    collect_output_weight_variables,
    collect_symbolic_weight_variables,
    compile_weight_mapping,
)

if TYPE_CHECKING:
    from wfomc.algo.core import AlgoOptions
    from wfomc.engine.features import FeatureSet
    from wfomc.problem import (
        CompiledBranchInstance,
        Domain,
        Problem,
    )
    from wfomc.reduction.reduced import DecoderSpec, ReducedProblem


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledReducedProblem:
    """Domain-free numeric branch consumed by a staged input builder."""

    reduced_problem: "ReducedProblem"
    sentence: object
    arithmetic: ArithmeticContext
    weights: dict[object, tuple[object, object]]
    feature_set: "FeatureSet"
    decoder_spec: "DecoderSpec"


@dataclass(frozen=True)
class CompiledSourceProblem:
    """Domain-free source formula and numeric weights for direct grounding."""

    problem: "Problem"
    arithmetic: ArithmeticContext
    weights: dict[object, tuple[object, object]]
    feature_set: "FeatureSet"


def compile_source_arithmetic(
    problem: "Problem",
    options: "AlgoOptions",
) -> tuple[ArithmeticContext, dict[object, tuple[object, object]]]:
    """Compile source weights without normalizing or reducing the formula."""

    output_symbols = collect_output_weight_variables(problem)
    solver_symbols = collect_symbolic_weight_variables(problem)
    backend = choose_arithmetic_backend(
        options.weight_options,
        symbolic_variables=solver_symbols,
    )
    arithmetic = ArithmeticContext(
        backend=backend,
        symbolic_variables=solver_symbols,
        output_symbols=output_symbols,
    )
    weights = compile_weight_mapping(dict(problem.weights), arithmetic)
    logger.info(
        "Prepared source arithmetic: backend=%s solver_symbols=%d output_symbols=%d",
        backend,
        len(solver_symbols),
        len(output_symbols),
    )
    return arithmetic, dict(sorted(weights.items(), key=lambda item: str(item[0])))


def compile_source_problem(
    problem: "Problem",
    options: "AlgoOptions",
) -> CompiledSourceProblem:
    """Compile a source problem without normalizing or binding a domain."""

    from wfomc.engine.features import analyze_features

    arithmetic, weights = compile_source_arithmetic(problem, options)
    return CompiledSourceProblem(
        problem=problem,
        arithmetic=arithmetic,
        weights=weights,
        feature_set=analyze_features(problem),
    )


def compile_reduced_problem(
    problem: "ReducedProblem",
    options: "AlgoOptions",
) -> CompiledReducedProblem:
    """Compile a domain-free reduced problem without concrete degree bounds."""

    from wfomc.fol import true

    sentence = problem.normal_form.qf_formula
    if sentence is None:
        sentence = true()
    output_symbols = collect_output_weight_variables(problem)
    solver_symbols = tuple(
        sorted(
            set(collect_symbolic_weight_variables(problem))
            | set(problem.internal_weight_symbols)
        )
    )
    if problem.internal_weight_symbols and options.weight_options.precision == "round":
        raise ArithmeticBackendError(
            "rounded arithmetic does not support cardinality marker variables; "
            "python-flint has no arb_mpoly backend"
        )
    backend = choose_arithmetic_backend(
        options.weight_options,
        symbolic_variables=solver_symbols,
    )
    arithmetic = ArithmeticContext(
        backend=backend,
        symbolic_variables=solver_symbols,
        output_symbols=output_symbols,
    )
    weights = compile_weight_mapping(dict(problem.weights), arithmetic)
    from wfomc.engine.features import analyze_features
    from wfomc.problem import CompiledBranchInstance

    feature_source = CompiledBranchInstance(
        sentence=sentence,
        arithmetic=arithmetic,
        weights=weights,
        evidence=problem.evidence,
    )
    return CompiledReducedProblem(
        reduced_problem=problem,
        sentence=sentence,
        arithmetic=arithmetic,
        weights=dict(sorted(weights.items(), key=lambda item: str(item[0]))),
        feature_set=analyze_features(
            feature_source,
            normal_form=problem.normal_form,
        ),
        decoder_spec=problem.decoder_spec,
    )


def instantiate_reduced_problem(
    compiled: CompiledReducedProblem,
    domain: "Domain",
) -> tuple["CompiledBranchInstance", object] | None:
    """Bind one compiled logical branch to a concrete domain."""

    reduced = compiled.reduced_problem
    if not reduced.applies(domain):
        return None
    limits = []
    for symbol, expression in reduced.internal_weight_degree_limits:
        value = expression.evaluate(domain.size)
        if value.denominator == 1 and value >= 0:
            limits.append((symbol, value.numerator))
    arithmetic = ArithmeticContext(
        backend=compiled.arithmetic.backend,
        symbolic_variables=compiled.arithmetic.symbolic_variables,
        output_symbols=compiled.arithmetic.output_symbols,
        degree_limits=tuple(limits),
    )
    weights = compile_weight_mapping(dict(compiled.weights), arithmetic)
    profile = (
        None
        if reduced.profile_constraint is None
        else reduced.profile_constraint.instantiate(domain.size)
    )
    from wfomc.problem import CompiledBranchInstance

    concrete = CompiledBranchInstance(
        sentence=compiled.sentence,
        arithmetic=arithmetic,
        domain=domain.elements,
        weights=dict(sorted(weights.items(), key=lambda item: str(item[0]))),
        evidence=reduced.evidence,
        profile_capacity_constraint=profile,
        circular_order_size=domain.circular_order_size,
    )
    return concrete, compiled.decoder_spec.instantiate(domain.size)


__all__ = [
    "CompiledReducedProblem",
    "CompiledSourceProblem",
    "compile_source_arithmetic",
    "compile_source_problem",
    "compile_reduced_problem",
    "instantiate_reduced_problem",
]
