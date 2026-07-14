"""Canonical WFOMC engine orchestration — pure dispatch, no algorithm branches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
import math
from time import perf_counter

from wfomc.algo.core import AlgoInput, AlgoName, AlgoOptions, PreparedBranch, algo_spec
from wfomc.engine.features import FeatureSet, analyze_features
from wfomc.engine.runtime import RuntimeContext, RuntimeOptions
from wfomc.problem import Problem
from wfomc.reduction import (
    ProblemWithDecoder,
    ReducedProblems,
)
from wfomc.result import WFOMCResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileArtifacts:
    parsed_problem: Problem
    feature_set: FeatureSet | None = None
    algo: AlgoName | None = None
    algo_options: AlgoOptions | None = None
    reduced_problem: ReducedProblems | None = None
    algo_inputs: tuple[AlgoInput, ...] = ()
    algo_input: AlgoInput | None = None


def analyze_problem(
    problem: Problem,
    *,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> CompileArtifacts:
    context = RuntimeContext.from_runtime(runtime)
    feature_set = context.cache.get_or_build(
        "features",
        _feature_key(problem),
        lambda: analyze_features(problem),
    )
    return CompileArtifacts(
        parsed_problem=problem,
        feature_set=feature_set,
    )


def compile_problem(
    problem: Problem,
    *,
    algo: AlgoName = AlgoName.STANDARD,
    options: AlgoOptions | None = None,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> CompileArtifacts:
    started = perf_counter()
    context = RuntimeContext.from_runtime(runtime)
    selected_algo = _require_algo_name(algo)
    analysis = analyze_problem(problem, runtime=context)
    if analysis.feature_set is None:
        raise RuntimeError("problem analysis did not produce features")
    spec = algo_spec(selected_algo)
    resolved_options = spec.resolve_options(analysis.feature_set, options)

    prepared = context.cache.get_or_build(
        "algo_inputs",
        _preparation_key(problem, selected_algo, resolved_options),
        lambda: spec.prepare(problem, resolved_options),
    )
    if not isinstance(prepared, tuple) or not all(
        isinstance(branch, PreparedBranch) for branch in prepared
    ):
        raise TypeError("algorithm prepare must return tuple[PreparedBranch, ...]")
    reduced = ReducedProblems(
        tuple(ProblemWithDecoder(branch.problem, branch.decoder) for branch in prepared)
    )
    algo_inputs = tuple(branch.algo_input for branch in prepared)
    logger.info(
        "Compiled problem: algo=%s domain=%d branches=%d elapsed_ms=%.3f",
        selected_algo.value,
        len(problem.domain),
        len(algo_inputs),
        (perf_counter() - started) * 1000,
    )

    return CompileArtifacts(
        parsed_problem=problem,
        feature_set=analysis.feature_set,
        algo=selected_algo,
        algo_options=resolved_options,
        reduced_problem=reduced,
        algo_inputs=algo_inputs,
        algo_input=algo_inputs[0] if algo_inputs else None,
    )


def solve(
    problem: Problem,
    *,
    algo: AlgoName = AlgoName.STANDARD,
    options: AlgoOptions | None = None,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> WFOMCResult:
    context = RuntimeContext.from_runtime(runtime)
    selected_algo = _require_algo_name(algo)
    result_key = _result_key(problem, algo=selected_algo, options=options)

    def compute() -> WFOMCResult:
        started = perf_counter()
        artifacts = compile_problem(
            problem,
            algo=selected_algo,
            options=options,
            runtime=context,
        )
        result = _run_reduced_problems(selected_algo, artifacts, context)
        logger.info(
            "Solved problem: algo=%s domain=%d branches=%d elapsed_ms=%.3f",
            selected_algo.value,
            len(problem.domain),
            len(artifacts.algo_inputs),
            (perf_counter() - started) * 1000,
        )
        return result

    return context.cache.get_or_build("results", result_key, compute)


def solve_uncached(
    problem: Problem,
    *,
    algo: AlgoName = AlgoName.STANDARD,
    options: AlgoOptions | None = None,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> WFOMCResult:
    context = RuntimeContext.from_runtime(runtime)
    artifacts = compile_problem(
        problem,
        algo=algo,
        options=options,
        runtime=context,
    )
    return _run_reduced_problems(artifacts.algo, artifacts, context)


def _require_algo_name(algo: AlgoName) -> AlgoName:
    if not isinstance(algo, AlgoName):
        raise TypeError(f"algo must be AlgoName, got {type(algo).__name__}")
    return algo


def _run(algo: AlgoName, algo_input: AlgoInput, context: RuntimeContext) -> WFOMCResult:
    spec = algo_spec(algo)
    result = spec.solve(algo_input, context)
    if not isinstance(result, WFOMCResult):
        raise TypeError(
            f"{algo.value} returned {type(result).__name__}; "
            "algorithms must return WFOMCResult"
        )
    return result


def _run_reduced_problems(
    algo: AlgoName | None,
    artifacts: CompileArtifacts,
    context: RuntimeContext,
) -> WFOMCResult:
    if algo is None:
        raise RuntimeError("compile artifacts do not specify an algorithm")
    if artifacts.reduced_problem is None:
        raise RuntimeError("compile artifacts do not contain reduced problems")
    if len(artifacts.reduced_problem.problems) != len(artifacts.algo_inputs):
        raise RuntimeError("reduced problem and algorithm input counts differ")

    total = None
    output_symbols: set[str] = set()
    for reduced_problem, algo_input in zip(
        artifacts.reduced_problem.problems,
        artifacts.algo_inputs,
    ):
        raw = _run(algo, algo_input, context).raw
        if (
            artifacts.feature_set is not None
            and artifacts.feature_set.has_linear_order
            and algo_input.include_order_factorial()
        ):
            raw *= algo_input.arithmetic.from_int(
                math.factorial(len(artifacts.parsed_problem.domain))
            )
        output_symbols.update(algo_input.arithmetic.output_symbols)
        decoded = reduced_problem.decoder(
            raw,
            arithmetic=algo_input.arithmetic,
            include_order_factorial=algo_input.include_order_factorial(),
        )
        total = decoded if total is None else total + decoded
    if total is None:
        raise RuntimeError("reduction produced no problems")
    return WFOMCResult(total, tuple(sorted(output_symbols)))


# ---------------------------------------------------------------------------
# Cache keys
#
# Cache keys serialize mixed dataclass/enums/FLINT values internally, but the
# public edge of each helper remains a source ``Problem``.
# ---------------------------------------------------------------------------


def _feature_key(problem: Problem) -> tuple[object, ...]:
    return ("features-v1", problem.cache_key_parts(include_domain_size=False))


def _preparation_key(
    problem: Problem,
    algo: AlgoName,
    options: AlgoOptions,
) -> tuple[object, ...]:
    return (
        "prepared-v1",
        algo.value,
        problem.cache_key_parts(include_domain_size=True),
        _options_key(options),
    )


def _result_key(
    problem: Problem,
    *,
    algo: AlgoName,
    options: AlgoOptions | None,
) -> tuple[object, ...]:
    return (
        "result-v1",
        algo.value,
        problem.cache_key_parts(include_domain_size=True),
        _options_key(options),
    )


def _strategy_option_key(value: object) -> object:
    if isinstance(value, Enum) and isinstance(value.value, str):
        return value.value
    return _object_key(value)


def _options_key(options: AlgoOptions | None) -> object:
    if options is None:
        return None
    return (
        _strategy_option_key(options.evidence_strategy),
        _strategy_option_key(options.existential_strategy),
        _strategy_option_key(options.linear_order_encoding),
        _object_key(options.weight_options),
    )


def _mapping_key(mapping: object) -> object:
    if not isinstance(mapping, dict):
        return _object_key(mapping)
    return tuple(sorted((_object_key(k), _object_key(v)) for k, v in mapping.items()))


def _object_key(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, dict):
        return _mapping_key(value)
    if isinstance(value, (tuple, list)):
        return tuple(_object_key(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_object_key(item) for item in value))
    if isinstance(value, Enum) and isinstance(value.value, str):
        return value.value
    return (type(value).__module__, type(value).__qualname__, str(value))


__all__ = [
    "CompileArtifacts",
    "analyze_problem",
    "compile_problem",
    "solve",
    "solve_uncached",
]
