"""Domain-free numeric compilation and concrete branch instantiation."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
import logging

from wfomc.algo.core import AlgoOptions
from wfomc.arithmetic import (
    ArithmeticBackend,
    ArithmeticContext,
    choose_arithmetic_backend,
)
from wfomc.cardinality_constraints import CardinalityConstraints
from wfomc.errors import ArithmeticBackendError
from wfomc.problem import (
    Domain,
    Problem,
)
from wfomc.stages import (
    CardinalityDecoderSpec,
    CompiledBranchInstance,
    CompiledReducedBranch,
    DecoderSpec,
    DivideDecoderSpec,
    FeatureSet,
    GroundingProblem,
    ReducedProblem,
)
from wfomc.weights import (
    collect_output_weight_variables,
    collect_symbolic_weight_variables,
    compile_weight_mapping,
)

logger = logging.getLogger(__name__)


def _compile_source_arithmetic(
    problem: Problem,
    options: AlgoOptions,
) -> tuple[ArithmeticContext, dict[object, tuple[object, object]]]:
    """Compile source weights without normalizing or reducing the formula."""

    output_symbols = collect_output_weight_variables(problem.weights)
    solver_symbols = collect_symbolic_weight_variables(problem.weights)
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


def compile_grounding_problem(
    problem: Problem,
    options: AlgoOptions,
    feature_set: FeatureSet,
) -> GroundingProblem:
    """Compile a source problem without normalizing or binding a domain."""

    arithmetic, weights = _compile_source_arithmetic(problem, options)
    return GroundingProblem(
        problem=problem,
        arithmetic=arithmetic,
        weights=weights,
        feature_set=feature_set,
    )


def compile_reduced_branch(
    problem: ReducedProblem,
    options: AlgoOptions,
) -> CompiledReducedBranch:
    """Compile a domain-free reduced problem without concrete degree bounds."""

    from wfomc.fol import true

    sentence = problem.normal_form.qf_formula
    if sentence is None:
        sentence = true()
    output_symbols = collect_output_weight_variables(
        problem.weights,
        problem.internal_weight_symbols,
    )
    solver_symbols = tuple(
        sorted(
            set(collect_symbolic_weight_variables(problem.weights))
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
        prefer_truncated_series=(
            backend is ArithmeticBackend.FMPQ_POLY
            and options.weight_options.exact_symbolic_backend == "auto"
        ),
    )
    weights = compile_weight_mapping(dict(problem.weights), arithmetic)
    for step in problem.decoder_spec.steps:
        if not isinstance(step, CardinalityDecoderSpec):
            continue
        for predicate, marker, exponent in step.predicate_markers:
            positive, negative = weights.get(
                predicate,
                (arithmetic.one(), arithmetic.one()),
            )
            weights[predicate] = (
                arithmetic.multiply(
                    positive,
                    arithmetic.power(arithmetic.symbol(marker), exponent),
                ),
                negative,
            )
    from wfomc.engine.features import analyze_reduced_features

    return CompiledReducedBranch(
        reduced_problem=problem,
        sentence=sentence,
        arithmetic=arithmetic,
        weights=dict(sorted(weights.items(), key=lambda item: str(item[0]))),
        feature_set=analyze_reduced_features(problem),
    )


def instantiate_reduced_branch(
    compiled: CompiledReducedBranch,
    domain: Domain,
) -> tuple[CompiledBranchInstance, Callable[..., object]] | None:
    """Bind one compiled logical branch to a concrete domain."""

    reduced = compiled.reduced_problem
    if domain.size < reduced.min_domain_size:
        return None
    limits = []
    for symbol, expression in reduced.internal_weight_degree_limits:
        value = expression.evaluate(domain.size)
        if value.denominator == 1 and value >= 0:
            limits.append((symbol, value.numerator))
    backend = compiled.arithmetic.backend
    if (
        compiled.arithmetic.prefer_truncated_series
        and backend is ArithmeticBackend.FMPQ_POLY
        and len(compiled.arithmetic.symbolic_variables) == 1
        and compiled.arithmetic.symbolic_variables[0]
        in {symbol for symbol, _limit in limits}
    ):
        backend = ArithmeticBackend.FMPQ_SERIES
    arithmetic = ArithmeticContext(
        backend=backend,
        symbolic_variables=compiled.arithmetic.symbolic_variables,
        output_symbols=compiled.arithmetic.output_symbols,
        degree_limits=tuple(limits),
        prefer_truncated_series=compiled.arithmetic.prefer_truncated_series,
    )
    weights = compile_weight_mapping(dict(compiled.weights), arithmetic)
    profile = (
        None
        if reduced.profile_constraint is None
        else reduced.profile_constraint.instantiate(domain.size)
    )
    concrete = CompiledBranchInstance(
        sentence=compiled.sentence,
        arithmetic=arithmetic,
        domain=domain.elements,
        weights=dict(sorted(weights.items(), key=lambda item: str(item[0]))),
        evidence=reduced.evidence,
        profile_capacity_constraint=profile,
        circular_order_size=domain.circular_order_size,
    )
    return concrete, _instantiate_decoder(reduced.decoder_spec, domain.size)


def _instantiate_decoder(
    decoder_spec: DecoderSpec,
    domain_size: int,
) -> Callable[..., object]:
    """Bind a data-only decoder specification to one domain size."""

    concrete_steps = []
    for step in decoder_spec.steps:
        if isinstance(step, DivideDecoderSpec):
            concrete_steps.append(("divide", step.coefficient.evaluate(domain_size)))
        elif isinstance(step, CardinalityDecoderSpec):
            concrete_steps.append(
                (
                    "cardinality",
                    CardinalityConstraints(
                        tuple(
                            constraint.instantiate(domain_size)
                            for constraint in step.constraints
                        )
                    ),
                    step.predicate_markers,
                    step.marker_encoding,
                )
            )
        else:
            raise TypeError(f"Unknown decoder step: {type(step).__name__}")

    def decode(result: object, **kwargs: object) -> object:
        arithmetic = kwargs.get("arithmetic")
        if not isinstance(arithmetic, ArithmeticContext):
            raise TypeError("reduced-problem decoder requires ArithmeticContext")
        value = result
        for concrete in reversed(concrete_steps):
            if concrete[0] == "divide":
                coefficient = Fraction(concrete[1])
                value = arithmetic.multiply(
                    value,
                    arithmetic.from_fraction(
                        coefficient.denominator,
                        coefficient.numerator,
                    ),
                )
            else:
                from wfomc.reduction.cardinality import (
                    decode_cardinality_result,
                )

                value = decode_cardinality_result(
                    value,
                    concrete[1],
                    concrete[2],
                    arithmetic,
                    marker_encoding=concrete[3],
                )
        return value

    return decode


__all__ = [
    "compile_reduced_branch",
    "compile_grounding_problem",
    "instantiate_reduced_branch",
]
