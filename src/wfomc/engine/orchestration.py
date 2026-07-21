"""Domain-separated WFOMC compilation, instantiation, and solving."""

from __future__ import annotations

from collections.abc import Hashable
from enum import Enum
import logging
import math
from time import perf_counter

from wfomc.algo.core import (
    AlgoInput,
    AlgoInputTemplate,
    AlgoName,
    AlgoOptions,
    AlgoSpec,
    GroundingInputTemplate,
    ReducedInputTemplate,
    SolveContext,
    algo_spec,
)
from wfomc.engine.artifacts import (
    CompiledProblem,
    ExecutionBranch,
    ProblemExecution,
)
from wfomc.engine.compilation import (
    compile_reduced_branch,
    compile_grounding_problem,
    instantiate_reduced_branch,
)
from wfomc.engine.features import analyze_problem_features
from wfomc.engine.runtime import RuntimeContext, RuntimeOptions
from wfomc.problem import (
    Domain,
    Problem,
    ProblemInstance,
)
from wfomc.stages import (
    CompiledReducedBranch,
    FeatureSet,
    GroundingProblem,
    ReducedProblem,
)
from wfomc.reduction import reduce_problem
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
        lambda: analyze_problem_features(source),
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
        branches = _compile_branches(
            source,
            features,
            spec,
            resolved_options,
        )
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
        execution_branches = []
        for branch_index, branch in enumerate(compiled.branches):
            input_key = (
                None
                if spec.input_template_key is None
                else spec.input_template_key(branch, domain)
            )
            def build_template() -> AlgoInputTemplate:
                return spec.build_input_template(
                    branch,
                    input_key,
                    compiled.algo_options,
                )

            cached_template = context.cache.get_or_build(
                "algo_input_templates",
                _input_template_key(
                    compiled,
                    branch_index,
                    input_key,
                ),
                build_template,
            )
            if not isinstance(
                cached_template,
                (ReducedInputTemplate, GroundingInputTemplate),
            ):
                raise TypeError(
                    f"{compiled.algo.value} input builder returned "
                    f"{type(cached_template).__name__}, expected an "
                    "algorithm input template"
                )
            input_template = cached_template
            context.cache.trim(
                "algo_input_templates",
                context.options.input_template_cache_size,
            )
            instantiated = _instantiate_execution_branch(
                branch,
                input_template,
                domain,
            )
            if instantiated is not None:
                execution_branches.append(instantiated)
        return ProblemExecution(
            compiled_problem=compiled,
            domain=domain,
            branches=tuple(execution_branches),
        )

    execution = context.cache.get_or_build("executions", key, build)
    context.cache.trim("executions", context.options.execution_cache_size)
    if not isinstance(execution, ProblemExecution):
        raise TypeError("execution cache returned an invalid value")
    logger.info(
        "Instantiated problem: algo=%s domain=%d branches=%d",
        compiled.algo.value,
        domain.size,
        len(execution.branches),
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
            len(execution.branches),
            (perf_counter() - started) * 1000,
        )
        return result

    result = context.cache.get_or_build("results", result_key, compute)
    context.cache.trim("results", context.options.result_cache_size)
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
    solve_context = SolveContext(
        ganak_path=context.options.propositional_ganak_path,
    )
    result = algo_spec(algo).solve(algo_input, solve_context)
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
    for branch in execution.branches:
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


def _compile_branches(
    problem: Problem,
    feature_set: FeatureSet,
    spec: AlgoSpec,
    options: AlgoOptions,
) -> tuple[CompiledReducedBranch | GroundingProblem, ...]:
    if not spec.uses_reduction:
        return (compile_grounding_problem(problem, options, feature_set),)
    reduced = reduce_problem(
        problem,
        evidence_strategy=options.evidence_strategy,
        existential_strategy=options.existential_strategy,
        lower_counting=spec.reduce_counting_quantifiers,
    )
    return (compile_reduced_branch(reduced, options),)


def _instantiate_execution_branch(
    branch: CompiledReducedBranch | GroundingProblem,
    input_template: AlgoInputTemplate,
    domain: Domain,
) -> ExecutionBranch | None:
    logical_problem: Problem | ReducedProblem
    if isinstance(branch, CompiledReducedBranch):
        if not isinstance(input_template, ReducedInputTemplate):
            raise TypeError(
                f"{type(branch).__name__} requires ReducedInputTemplate, got "
                f"{type(input_template).__name__}"
            )
        instantiated = instantiate_reduced_branch(branch, domain)
        if instantiated is None:
            return None
        concrete, decoder = instantiated
        algo_input = input_template.instantiate(concrete)
        logical_problem = branch.reduced_problem
    elif isinstance(branch, GroundingProblem):
        if not isinstance(input_template, GroundingInputTemplate):
            raise TypeError(
                f"{type(branch).__name__} requires GroundingInputTemplate, got "
                f"{type(input_template).__name__}"
            )
        algo_input = input_template.instantiate(domain)
        logical_problem = branch.problem
        decoder = _identity_decoder
    else:
        raise TypeError(f"Unknown compiled branch: {type(branch).__name__}")
    if not isinstance(algo_input, AlgoInput):
        raise TypeError(
            f"{type(input_template).__name__}.instantiate() returned "
            f"{type(algo_input).__name__}, expected AlgoInput"
        )
    return ExecutionBranch(logical_problem, algo_input, decoder)


def _identity_decoder(result: object, **_: object) -> object:
    return result


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
    input_key: Hashable,
) -> tuple[object, ...]:
    return (
        "algo-input-template-v2",
        _compilation_key(
            compiled.problem,
            compiled.algo,
            compiled.algo_options,
        ),
        branch_index,
        input_key,
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
