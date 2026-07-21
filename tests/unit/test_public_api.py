from __future__ import annotations

from wfomc import (
    AlgoName,
    AlgoOptions,
    BoundaryProfileOptions,
    CompiledProblem,
    Domain,
    EvidenceStrategy,
    ExistentialStrategy,
    LinearOrderEncoding,
    Problem,
    ProblemExecution,
    ProblemInstance,
    RuntimeContext,
    RuntimeOptions,
    WFOMCResult,
    WeightOptions,
    compile_problem,
    instantiate_problem,
    parse_problem,
    parse_problem_file,
    solve,
)


def test_documented_top_level_api_is_importable():
    """Protect the supported Python entry points without freezing all exports."""

    assert AlgoOptions().weight_options == WeightOptions()
    assert AlgoOptions().boundary_profile_options == BoundaryProfileOptions()
    assert isinstance(LinearOrderEncoding.PIN.value, str)
    assert isinstance(EvidenceStrategy.CCS.value, str)
    assert isinstance(ExistentialStrategy.COUNTING.value, str)
    assert all(
        item is not None
        for item in (
            AlgoName,
            CompiledProblem,
            Domain,
            Problem,
            ProblemExecution,
            ProblemInstance,
            RuntimeContext,
            RuntimeOptions,
            WFOMCResult,
            compile_problem,
            instantiate_problem,
            parse_problem,
            parse_problem_file,
            solve,
        )
    )
