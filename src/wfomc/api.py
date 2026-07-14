"""Public API facade for the framework runtime."""

from __future__ import annotations


from .algo import (
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    EvidenceStrategy,
    ExistentialStrategy,
)
from .engine import CompileArtifacts
from .engine import compile_problem as engine_compile_problem
from .engine import solve as engine_solve
from .engine.runtime import RuntimeContext, RuntimeOptions
from .problem import Problem
from .result import WFOMCResult


def compile_problem(
    problem: Problem,
    *,
    algo: AlgoName = AlgoName.STANDARD,
    options: AlgoOptions | None = None,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> CompileArtifacts:
    """Compile a WFOMC problem into algorithm-owned input artifacts."""

    return engine_compile_problem(
        problem,
        algo=algo,
        options=options,
        runtime=runtime,
    )


def solve(
    problem: Problem,
    *,
    algo: AlgoName = AlgoName.STANDARD,
    options: AlgoOptions | None = None,
    runtime: RuntimeContext | RuntimeOptions | None = None,
) -> WFOMCResult:
    """Solve a problem through the framework algorithm registry."""

    return engine_solve(
        problem,
        algo=algo,
        options=options,
        runtime=runtime,
    )


__all__ = [
    "AlgoName",
    "AlgoMaturity",
    "AlgoOptions",
    "CompileArtifacts",
    "EvidenceStrategy",
    "ExistentialStrategy",
    "Problem",
    "RuntimeContext",
    "RuntimeOptions",
    "WFOMCResult",
    "compile_problem",
    "solve",
]
