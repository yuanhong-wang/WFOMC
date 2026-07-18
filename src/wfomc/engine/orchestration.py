"""Domain-separated WFOMC compilation, instantiation, and solving."""

from __future__ import annotations

from enum import Enum
import logging
import math
from time import perf_counter

from wfomc.algo.core import AlgoInput, AlgoName, AlgoOptions, algo_spec
from wfomc.engine.features import FeatureSet, analyze_features
from wfomc.engine.runtime import RuntimeContext, RuntimeOptions
from wfomc.problem import (
    CompiledProblem,
    Domain,
    Problem,
    ProblemExecution,
    ProblemInstance,
)
from wfomc.result import WFOMCResult


logger = logging.getLogger(__name__)


def analyze_problem(
    problem: Problem | ProblemInstance,
    *,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> FeatureSet:
    """Analyze a logical problem independently of every concrete domain."""

    source = _source_problem(problem)
    context = RuntimeContext.from_runtime(runtime)
    feature_set = context.cache.get_or_build(
        "features",
        _feature_key(source),
        lambda: analyze_features(source),
    )
    if not isinstance(feature_set, FeatureSet):
        raise TypeError("feature cache returned a non-FeatureSet value")
    return feature_set


def compile_problem(
    problem: Problem | ProblemInstance,
    *,
    algo: AlgoName = AlgoName.STANDARD,
    options: AlgoOptions | None = None,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> CompiledProblem:
    """Compile a reusable logical problem without binding a domain."""

    started = perf_counter()
    source = _source_problem(problem)
    context = RuntimeContext.from_runtime(runtime)
    selected_algo = _require_algo_name(algo)
    features = analyze_problem(source, runtime=context)
    spec = algo_spec(selected_algo)
    resolved_options = spec.resolve_options(features, options)
    key = _compilation_key(source, selected_algo, resolved_options)

    def build() -> CompiledProblem:
        branches = spec.compile_branches(source, resolved_options)
        if not isinstance(branches, tuple):
            raise TypeError("compile_branches must return a tuple")
        return CompiledProblem(
            problem=source,
            feature_set=features,
            algo=selected_algo,
            algo_options=resolved_options,
            branches=branches,
        )

    compiled = context.cache.get_or_build("compiled_problems", key, build)
    if not isinstance(compiled, CompiledProblem):
        raise TypeError("compiled-problem cache returned an invalid value")
    logger.info(
        "Compiled domain-free problem: algo=%s branches=%d elapsed_ms=%.3f",
        selected_algo.value,
        len(compiled.branches),
        (perf_counter() - started) * 1000,
    )
    return compiled


def instantiate_problem(
    compiled: CompiledProblem,
    domain: Domain,
    *,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> ProblemExecution:
    """Instantiate one reusable compilation for a concrete finite domain."""

    if not isinstance(compiled, CompiledProblem):
        raise TypeError("compiled must be a CompiledProblem")
    if not isinstance(domain, Domain):
        raise TypeError("domain must be a Domain")
    _validate_domain(compiled.problem, domain)
    context = RuntimeContext.from_runtime(runtime)
    key = _execution_key(compiled, domain)

    def build() -> ProblemExecution:
        spec = algo_spec(compiled.algo)
        prepared_branches = []
        for branch_index, branch in enumerate(compiled.branches):
            if spec.branch_applies is not None and not spec.branch_applies(
                branch, domain
            ):
                continue
            input_variant = (
                None
                if spec.input_template_variant is None
                else spec.input_template_variant(branch, domain)
            )
            input_template = context.cache.get_or_build(
                "algo_input_templates",
                _input_template_key(
                    compiled,
                    branch_index,
                    input_variant,
                ),
                lambda branch=branch, input_variant=input_variant: (
                    spec.build_input_template(
                        branch,
                        input_variant,
                        compiled.algo_options,
                    )
                ),
            )
            instantiated = spec.instantiate_branch(
                branch,
                input_template,
                domain,
                compiled.algo_options,
            )
            if instantiated is not None:
                prepared_branches.append(instantiated)
        prepared = tuple(prepared_branches)
        return ProblemExecution(
            compiled_problem=compiled,
            domain=domain,
            prepared_branches=prepared,
        )

    execution = context.cache.get_or_build("executions", key, build)
    context.cache.trim("executions", context.options.execution_cache_size)
    if not isinstance(execution, ProblemExecution):
        raise TypeError("execution cache returned an invalid value")
    logger.info(
        "Instantiated problem: algo=%s domain=%d branches=%d",
        compiled.algo.value,
        domain.size,
        len(execution.prepared_branches),
    )
    return execution


def solve(
    problem: Problem | ProblemInstance | CompiledProblem,
    domain: Domain | None = None,
    *,
    algo: AlgoName = AlgoName.STANDARD,
    options: AlgoOptions | None = None,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> WFOMCResult:
    """Compile if needed, instantiate one domain, and solve."""

    context = RuntimeContext.from_runtime(runtime)
    compiled, concrete_domain = _resolve_solve_target(
        problem,
        domain,
        algo=algo,
        options=options,
        runtime=context,
    )
    result_key = _result_key(compiled, concrete_domain)

    def compute() -> WFOMCResult:
        started = perf_counter()
        execution = instantiate_problem(
            compiled,
            concrete_domain,
            runtime=context,
        )
        result = _run_execution(execution, context)
        logger.info(
            "Solved problem: algo=%s domain=%d branches=%d elapsed_ms=%.3f",
            compiled.algo.value,
            concrete_domain.size,
            len(execution.prepared_branches),
            (perf_counter() - started) * 1000,
        )
        return result

    result = context.cache.get_or_build("results", result_key, compute)
    if not isinstance(result, WFOMCResult):
        raise TypeError("result cache returned an invalid value")
    return result


def _resolve_solve_target(
    target: Problem | ProblemInstance | CompiledProblem,
    domain: Domain | None,
    *,
    algo: AlgoName,
    options: AlgoOptions | None,
    runtime: RuntimeContext,
) -> tuple[CompiledProblem, Domain]:
    if isinstance(target, CompiledProblem):
        if domain is None:
            raise TypeError("solve(compiled, ...) requires a Domain")
        if options is not None:
            raise TypeError("options are already fixed by CompiledProblem")
        if algo is not AlgoName.STANDARD and algo is not target.algo:
            raise TypeError("algo is already fixed by CompiledProblem")
        return target, domain
    if isinstance(target, ProblemInstance):
        if domain is not None and domain != target.domain:
            raise TypeError("domain was provided twice with different values")
        domain = target.domain
        target = target.problem
    if not isinstance(target, Problem):
        raise TypeError(
            "solve target must be Problem, ProblemInstance, or CompiledProblem"
        )
    if domain is None:
        raise TypeError("solve(problem, ...) requires a Domain")
    return (
        compile_problem(
            target,
            algo=algo,
            options=options,
            runtime=runtime,
        ),
        domain,
    )


def _run(
    algo: AlgoName,
    algo_input: AlgoInput,
    context: RuntimeContext,
) -> WFOMCResult:
    result = algo_spec(algo).solve(algo_input, context)
    if not isinstance(result, WFOMCResult):
        raise TypeError(
            f"{algo.value} returned {type(result).__name__}; "
            "algorithms must return WFOMCResult"
        )
    return result


def _run_execution(
    execution: ProblemExecution,
    context: RuntimeContext,
) -> WFOMCResult:
    compiled = execution.compiled_problem
    total = None
    output_symbols: set[str] = set()
    for branch in execution.prepared_branches:
        algo_input = branch.algo_input
        raw = _run(compiled.algo, algo_input, context).raw
        if (
            compiled.feature_set.has_linear_order
            and algo_input.include_order_factorial()
        ):
            raw *= algo_input.arithmetic.from_int(math.factorial(execution.domain.size))
        output_symbols.update(algo_input.arithmetic.output_symbols)
        decoded = branch.decoder(
            raw,
            arithmetic=algo_input.arithmetic,
            include_order_factorial=algo_input.include_order_factorial(),
        )
        decoded = algo_input.arithmetic.project_to_output(decoded)
        total = decoded if total is None else total + decoded
    if total is None:
        raise RuntimeError("reduction produced no applicable problems")
    return WFOMCResult(total, tuple(sorted(output_symbols)))


def _validate_domain(problem: Problem, domain: Domain) -> None:
    missing = problem.required_domain_constants() - domain.elements
    if missing:
        raise ValueError(
            "Domain does not contain referenced constants: "
            + ", ".join(sorted(map(str, missing)))
        )


def _source_problem(problem: Problem | ProblemInstance) -> Problem:
    if isinstance(problem, ProblemInstance):
        return problem.problem
    if isinstance(problem, Problem):
        return problem
    raise TypeError("problem must be Problem or ProblemInstance")


def _require_algo_name(algo: AlgoName) -> AlgoName:
    if not isinstance(algo, AlgoName):
        raise TypeError(f"algo must be AlgoName, got {type(algo).__name__}")
    return algo


def _feature_key(problem: Problem) -> tuple[object, ...]:
    return ("features-v2", problem.cache_key_parts())


def _compilation_key(
    problem: Problem,
    algo: AlgoName,
    options: AlgoOptions,
) -> tuple[object, ...]:
    return (
        "compiled-v2",
        algo.value,
        problem.cache_key_parts(),
        _options_key(options),
    )


def _execution_key(
    compiled: CompiledProblem,
    domain: Domain,
) -> tuple[object, ...]:
    return (
        "execution-v2",
        _compilation_key(
            compiled.problem,
            compiled.algo,
            compiled.algo_options,
        ),
        domain.cache_key_parts(),
    )


def _input_template_key(
    compiled: CompiledProblem,
    branch_index: int,
    input_variant: object,
) -> tuple[object, ...]:
    return (
        "algo-input-template-v2",
        _compilation_key(
            compiled.problem,
            compiled.algo,
            compiled.algo_options,
        ),
        branch_index,
        input_variant,
    )


def _result_key(
    compiled: CompiledProblem,
    domain: Domain,
) -> tuple[object, ...]:
    return (
        "result-v2",
        _execution_key(compiled, domain),
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


def _object_key(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, dict):
        return tuple(sorted((_object_key(k), _object_key(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_object_key(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_object_key(item) for item in value))
    if isinstance(value, Enum) and isinstance(value.value, str):
        return value.value
    return (type(value).__module__, type(value).__qualname__, str(value))


__all__ = [
    "analyze_problem",
    "compile_problem",
    "instantiate_problem",
    "solve",
]
