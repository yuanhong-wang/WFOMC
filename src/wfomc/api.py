"""Public API facade for the framework runtime."""

from __future__ import annotations


from .algo import (
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    EvidenceStrategy,
    ExistentialStrategy,
    LinearOrderEncoding,
)
from .engine import compile_problem as engine_compile_problem
from .engine import instantiate_problem as engine_instantiate_problem
from .engine import solve as engine_solve
from .engine.runtime import RuntimeContext, RuntimeOptions
from .problem import (
    CompiledProblem,
    Domain,
    Problem,
    ProblemExecution,
    ProblemInstance,
)
from .result import WFOMCResult
from .weights import WeightOptions


def compile_problem(
    problem: Problem | ProblemInstance,
    *,
    algo: AlgoName = AlgoName.STANDARD,
    options: AlgoOptions | None = None,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> CompiledProblem:
    """Compile a reusable WFOMC problem without binding a domain."""

    return engine_compile_problem(
        problem,
        algo=algo,
        options=options,
        runtime=runtime,
    )


def instantiate_problem(
    compiled: CompiledProblem,
    domain: Domain,
    *,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> ProblemExecution:
    """Instantiate one compiled problem for a concrete domain."""

    return engine_instantiate_problem(compiled, domain, runtime=runtime)


def solve(
    problem: Problem | ProblemInstance | CompiledProblem,
    domain: Domain | None = None,
    *,
    algo: AlgoName = AlgoName.STANDARD,
    options: AlgoOptions | None = None,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> WFOMCResult:
    """Solve a problem through the framework algorithm registry."""

    return engine_solve(
        problem,
        domain,
        algo=algo,
        options=options,
        runtime=runtime,
    )


__all__ = [
    "AlgoName",
    "AlgoMaturity",
    "AlgoOptions",
    "CompiledProblem",
    "Domain",
    "EvidenceStrategy",
    "ExistentialStrategy",
    "LinearOrderEncoding",
    "Problem",
    "ProblemExecution",
    "ProblemInstance",
    "RuntimeContext",
    "RuntimeOptions",
    "WFOMCResult",
    "WeightOptions",
    "compile_problem",
    "instantiate_problem",
    "solve",
]
