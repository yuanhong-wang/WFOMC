from __future__ import annotations

from dataclasses import replace

import pytest

from wfomc.algo import (
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    algo_spec,
)
from wfomc.algo.fast.input import OptimizedCellGraphInput
from wfomc.algo.incremental.input import OrderedCellGraphInput
from wfomc.algo.propositional.input import GroundCNFInput
from wfomc.algo.standard.input import StandardInput
from wfomc.algo.core import GroundingInputTemplate, ReducedInputTemplate
from wfomc.engine import (
    analyze_problem,
    compile_problem,
    instantiate_problem,
    solve,
)
from wfomc.engine.artifacts import (
    CompiledProblem,
    ExecutionBranch,
    ProblemExecution,
)
from wfomc.engine.runtime import RuntimeContext
from wfomc.errors import UnsupportedFeatureError
from wfomc.evidence import Evidence, GroundUnaryLiteral, UnaryEvidence
from wfomc.fol import CountingQuantifier, FOLContext, walk
from wfomc.fol.grounding import LinearOrderEncoding
from wfomc.options import EvidenceStrategy
from wfomc.parser import parse_problem_file, parse_problem
from wfomc.problem import Domain
from wfomc.stages import FeatureSet, GroundingProblem


def test_analyze_problem_returns_domain_free_features():
    instance = parse_problem_file("models/unary_evidence/evidence-only.wfomcs")

    features = analyze_problem(instance.problem)

    assert isinstance(features, FeatureSet)
    assert features.has_unary_evidence


def test_algorithm_specs_declare_maturity():
    assert algo_spec(AlgoName.STANDARD).maturity is AlgoMaturity.STABLE
    assert algo_spec(AlgoName.PROPOSITIONAL).maturity is AlgoMaturity.BETA
    assert algo_spec(AlgoName.BOUNDED_TREEWIDTH).maturity is AlgoMaturity.UNAVAILABLE


def test_standard_compiles_domain_free_branch_and_instantiates_input_later():
    instance = parse_problem_file("models/unary_evidence/evidence-only.wfomcs")

    compiled = compile_problem(instance.problem, algo=AlgoName.STANDARD)

    assert isinstance(compiled, CompiledProblem)
    assert compiled.problem is instance.problem
    assert compiled.branches
    execution = instantiate_problem(compiled, instance.domain)
    assert isinstance(execution, ProblemExecution)
    assert isinstance(execution.branches[0], ExecutionBranch)
    assert isinstance(execution.algo_input, StandardInput)


def test_every_registered_algorithm_uses_staged_contract():
    for algo in AlgoName:
        spec = algo_spec(algo)
        assert callable(spec.build_input_template)
        assert isinstance(spec.uses_reduction, bool)
        assert isinstance(spec.reduce_counting_quantifiers, bool)


@pytest.mark.parametrize(
    "algo",
    (
        AlgoName.STANDARD,
        AlgoName.FAST,
        AlgoName.FASTV2,
        AlgoName.INCREMENTAL,
        AlgoName.INCREMENTAL3,
        AlgoName.RECURSIVE,
        AlgoName.PROPOSITIONAL,
        AlgoName.PROPOSITIONAL_REDUCED,
    ),
)
def test_staged_algorithm_reuses_input_template_across_domains(algo):
    instance = parse_problem_file("models/2-colored-graph.wfomcs")
    runtime = RuntimeContext()
    compiled = compile_problem(instance.problem, algo=algo, runtime=runtime)

    small = instantiate_problem(compiled, Domain.of_size(2), runtime=runtime)
    large = instantiate_problem(compiled, Domain.of_size(3), runtime=runtime)

    assert small.domain.size == 2
    assert large.domain.size == 3
    stats = runtime.cache.stats()
    assert stats.misses["algo_input_templates"] == 1
    assert stats.hits["algo_input_templates"] == 1
    template = next(iter(runtime.cache.algo_input_templates.values()))
    expected_template_type = (
        GroundingInputTemplate
        if algo is AlgoName.PROPOSITIONAL
        else ReducedInputTemplate
    )
    assert isinstance(template, expected_template_type)


def test_incremental3_rebuilds_template_only_when_counting_state_changes():
    instance = parse_problem(
        r"""
\forall X: (\exists_=3 Y: R(X,Y))
domain = 3
"""
    )
    runtime = RuntimeContext()
    compiled = compile_problem(
        instance.problem,
        algo=AlgoName.INCREMENTAL3,
        runtime=runtime,
    )

    instantiate_problem(compiled, Domain.of_size(2), runtime=runtime)
    instantiate_problem(compiled, Domain.of_size(3), runtime=runtime)
    instantiate_problem(compiled, Domain.of_size(4), runtime=runtime)

    stats = runtime.cache.stats()
    assert stats.misses["algo_input_templates"] == 2
    assert stats.hits["algo_input_templates"] == 1


@pytest.mark.parametrize("algo", (AlgoName.FAST, AlgoName.FASTV2))
def test_fast_compiles_reusable_branches_and_instantiates_concrete_input(algo):
    instance = parse_problem_file("models/2-colored-graph.wfomcs")

    compiled = compile_problem(instance.problem, algo=algo)
    execution = instantiate_problem(compiled, instance.domain)

    assert compiled.branches
    assert isinstance(execution.algo_input, OptimizedCellGraphInput)
    assert execution.algo_input.domain_size == instance.domain.size


def test_fast_input_template_is_reused_but_operations_are_fresh_per_domain():
    instance = parse_problem_file("models/2-colored-graph.wfomcs")
    runtime = RuntimeContext()
    compiled = compile_problem(
        instance.problem,
        algo=AlgoName.FASTV2,
        runtime=runtime,
    )
    small = Domain.of_size(2)
    large = Domain.of_size(3)

    small_execution = instantiate_problem(compiled, small, runtime=runtime)
    large_execution = instantiate_problem(compiled, large, runtime=runtime)

    stats = runtime.cache.stats()
    assert stats.misses["algo_input_templates"] == 1
    assert stats.hits["algo_input_templates"] == 1
    small_component = small_execution.algo_input.components[0]
    large_component = large_execution.algo_input.components[0]
    assert small_component.cliques == large_component.cliques
    assert small_component.weight_operations is not large_component.weight_operations


@pytest.mark.parametrize("algo", (AlgoName.FAST, AlgoName.FASTV2))
def test_one_fast_compilation_instantiates_domain_sized_counting_reduction(algo):
    instance = parse_problem(
        r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 1
"""
    )
    compiled = compile_problem(instance.problem, algo=algo)

    for size in (1, 2, 3):
        domain = Domain.of_size(size)
        assert solve(compiled, domain) == size**size
        assert (
            solve(
                instance.problem,
                domain,
                algo=AlgoName.STANDARD,
            )
            == size**size
        )


@pytest.mark.parametrize("algo", (AlgoName.FAST, AlgoName.FASTV2))
def test_unary_evidence_uses_one_branch_and_fast_input_structure_variants(algo):
    fol = FOLContext()
    predicate = fol.predicate("P", 1)
    observed = fol.constant("a")
    problem = parse_problem(r"\forall X: (P(X) | ~P(X))").problem
    problem = replace(
        problem,
        evidence=Evidence(
            unary=UnaryEvidence((GroundUnaryLiteral(predicate, observed, True),))
        ),
    )
    runtime = RuntimeContext()
    compiled = compile_problem(
        problem,
        algo=algo,
        options=AlgoOptions(
            evidence_strategy=EvidenceStrategy.LIFTED_PROFILES,
        ),
        runtime=runtime,
    )

    assert len(compiled.branches) == 1
    assert runtime.cache.stats().sizes["algo_input_templates"] == 0

    closed = Domain(frozenset((observed,)))
    open_domain = Domain(frozenset((observed, fol.constant("b"), fol.constant("c"))))
    closed_execution = instantiate_problem(compiled, closed, runtime=runtime)
    open_execution = instantiate_problem(compiled, open_domain, runtime=runtime)

    assert sum(
        len(component.cells) for component in closed_execution.algo_input.components
    ) < sum(len(component.cells) for component in open_execution.algo_input.components)
    assert runtime.cache.stats().misses["algo_input_templates"] == 2
    assert solve(compiled, closed, runtime=runtime) == 1
    assert solve(compiled, open_domain, runtime=runtime) == 4


def test_fast_ccs_selects_input_structure_without_reduction_branches():
    fol = FOLContext()
    predicate = fol.predicate("P", 1)
    observed = fol.constant("a")
    problem = replace(
        parse_problem(r"\forall X: (P(X) | ~P(X))").problem,
        evidence=Evidence(
            unary=UnaryEvidence((GroundUnaryLiteral(predicate, observed, True),))
        ),
    )
    runtime = RuntimeContext()
    compiled = compile_problem(
        problem,
        algo=AlgoName.FAST,
        options=AlgoOptions(evidence_strategy=EvidenceStrategy.CCS),
        runtime=runtime,
    )

    assert len(compiled.branches) == 1
    assert (
        solve(
            compiled,
            Domain(frozenset((observed,))),
            runtime=runtime,
        )
        == 1
    )
    assert (
        solve(
            compiled,
            Domain(frozenset((observed, fol.constant("b")))),
            runtime=runtime,
        )
        == 2
    )
    stats = runtime.cache.stats()
    assert stats.misses["algo_input_templates"] == 2


@pytest.mark.parametrize(
    ("predicate", "is_circular"),
    (("PRED", False), ("CIRCULAR_PRED", True)),
)
def test_order_features_drive_incremental_input(
    predicate: str,
    is_circular: bool,
):
    instance = parse_problem(
        rf"""
\forall X: (\forall Y: ({predicate}(X,Y) | ~{predicate}(X,Y)))
domain = 2
"""
    )
    compiled = compile_problem(instance.problem, algo=AlgoName.INCREMENTAL)
    execution = instantiate_problem(compiled, instance.domain)

    assert isinstance(execution.algo_input, OrderedCellGraphInput)
    assert execution.algo_input.predecessor_orders == (1,)
    assert execution.algo_input.has_circular_predecessor is is_circular


def test_circular_order_size_belongs_to_domain_and_reaches_incremental_input():
    instance = parse_problem(
        r"""
\forall X: (\forall Y: (CIRCULAR_PRED(X,Y) | ~CIRCULAR_PRED(X,Y)))
domain = 3
"""
    )
    domain = replace(instance.domain, circular_order_size=2)
    compiled = compile_problem(instance.problem, algo=AlgoName.INCREMENTAL)

    execution = instantiate_problem(compiled, domain)

    assert isinstance(execution.algo_input, OrderedCellGraphInput)
    assert execution.algo_input.circle_len == 2


def test_propositional_inputs_are_created_only_during_instantiation():
    instance = parse_problem_file("models/unary_evidence/evidence-only.wfomcs")
    compiled = compile_problem(instance.problem, algo=AlgoName.PROPOSITIONAL)

    assert isinstance(compiled.branches[0], GroundingProblem)
    execution = instantiate_problem(compiled, instance.domain)

    assert isinstance(execution.algo_input, GroundCNFInput)
    assert execution.algo_input.cnf


def test_direct_and_reduced_propositional_modes_keep_distinct_formulas():
    instance = parse_problem(
        r"""
\forall X: (\exists_=1 Y: R(X,Y))
domain = 2
"""
    )
    assert any(
        isinstance(node, CountingQuantifier) for node in walk(instance.problem.sentence)
    )

    direct = instantiate_problem(
        compile_problem(instance.problem, algo=AlgoName.PROPOSITIONAL),
        instance.domain,
    )
    reduced = instantiate_problem(
        compile_problem(instance.problem, algo=AlgoName.PROPOSITIONAL_REDUCED),
        instance.domain,
    )

    assert isinstance(direct.algo_input, GroundCNFInput)
    assert isinstance(reduced.algo_input, GroundCNFInput)


def test_propositional_order_option_is_fixed_by_compilation():
    instance = parse_problem_file("models/linear_order/head-middle-tail.wfomcs")
    compiled = compile_problem(
        instance.problem,
        algo=AlgoName.PROPOSITIONAL,
        options=AlgoOptions(linear_order_encoding=LinearOrderEncoding.AXIOMS),
    )

    execution = instantiate_problem(compiled, instance.domain)

    assert execution.algo_input.linear_order_encoding is LinearOrderEncoding.AXIOMS
    assert not execution.algo_input.include_order_factorial()


def test_unsupported_features_are_rejected_during_domain_free_compilation():
    instance = parse_problem_file("models/linear_order/head-middle-tail.wfomcs")

    with pytest.raises(UnsupportedFeatureError, match="linear order"):
        compile_problem(instance.problem, algo=AlgoName.STANDARD)

    with pytest.raises(UnsupportedFeatureError, match="linear order"):
        solve(instance, algo=AlgoName.STANDARD)


def test_instantiation_validates_named_constants_against_domain():
    instance = parse_problem(
        r"""
\forall X: P(a)
domain = {a}
"""
    )
    compiled = compile_problem(instance.problem, algo=AlgoName.FAST)

    with pytest.raises(ValueError, match="referenced constants"):
        instantiate_problem(compiled, Domain.of_size(1))


def test_compile_cache_reuses_same_domain_free_artifact():
    instance = parse_problem_file("models/2-colored-graph.wfomcs")
    runtime = RuntimeContext()

    first = compile_problem(instance.problem, algo=AlgoName.FAST, runtime=runtime)
    second = compile_problem(instance.problem, algo=AlgoName.FAST, runtime=runtime)

    assert second is first
    stats = runtime.cache.stats()
    assert stats.misses["compiled_problems"] == 1
    assert stats.hits["compiled_problems"] == 1
