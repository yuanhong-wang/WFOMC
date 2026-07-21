"""Engine-owned compilation and execution artifacts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from wfomc.algo.core import AlgoInput, AlgoName, AlgoOptions
from wfomc.problem import (
    Domain,
    Problem,
)
from wfomc.stages import (
    CompiledReducedBranch,
    FeatureSet,
    GroundingProblem,
    ReducedProblem,
)


@dataclass(frozen=True)
class CompiledProblem:
    """Domain-free compilation selected for one algorithm and option set."""

    problem: Problem
    feature_set: FeatureSet
    algo: AlgoName
    algo_options: AlgoOptions
    branches: tuple[CompiledReducedBranch | GroundingProblem, ...] = ()


@dataclass(frozen=True)
class ExecutionBranch:
    """One concrete algorithm input and its branch-local result decoder."""

    problem: Problem | ReducedProblem
    algo_input: AlgoInput
    decoder: Callable[..., object]


@dataclass(frozen=True)
class ProblemExecution:
    """One concrete-domain instantiation of a reusable compilation."""

    compiled_problem: CompiledProblem
    domain: Domain
    branches: tuple[ExecutionBranch, ...]

    @property
    def algo_input(self) -> AlgoInput | None:
        return self.branches[0].algo_input if self.branches else None


__all__ = [
    "CompiledProblem",
    "ExecutionBranch",
    "ProblemExecution",
]
