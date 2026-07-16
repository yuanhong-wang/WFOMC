"""Shared algorithm contracts."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, Protocol

from wfomc.arithmetic import (
    ArithmeticBackend,
    ArithmeticContext,
    choose_arithmetic_backend,
)
from wfomc.errors import (
    ArithmeticBackendError,
    UnsupportedFeatureError,
)
from wfomc.weights import (
    WeightOptions,
    collect_output_weight_variables,
    collect_symbolic_weight_variables,
    compile_weight_mapping,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from wfomc.engine.features import FeatureSet
    from wfomc.engine.runtime import RuntimeContext
    from wfomc.fol.grounding import LinearOrderEncoding
    from wfomc.problem import CompiledProblem, Problem, ReducedProblem
    from wfomc.reduction import ProblemWithDecoder
    from wfomc.result import WFOMCResult


class EvidenceStrategy(Enum):
    """Algorithm strategy for consuming unary evidence."""

    NONE = "none"
    CCS = "ccs"
    LIFTED_PROFILES = "lifted-profiles"
    GROUND_UNITS = "ground-units"
    PROFILE_CAPACITY_DP = "profile-capacity-dp"
    TREE_DECOMPOSITION_FACTORS = "tree-decomposition-factors"

    def __str__(self) -> str:
        return self.value


class ExistentialStrategy(Enum):
    """Reduction used for existentially quantified normal-form sections."""

    COUNTING = "counting"
    SKOLEM = "skolem"
    GROUND = "ground"

    def __str__(self) -> str:
        return self.value


SupportedEvidence = tuple[EvidenceStrategy, ...]


class DefaultEvidenceStrategy(Protocol):
    """Choose the default evidence strategy from analyzed source features."""

    def __call__(
        self,
        features: "FeatureSet",
        linear_order_encoding: "LinearOrderEncoding",
    ) -> EvidenceStrategy: ...


class SolveFn(Protocol):
    """Protocol for ``spec.solve``: ``AlgoInput × RuntimeContext|None → WFOMCResult``."""

    def __call__(
        self,
        algo_input: "AlgoInput",
        context: "RuntimeContext | None",
    ) -> "WFOMCResult": ...


class AlgoName(Enum):
    """Stable public names for WFOMC algorithms."""

    STANDARD = "standard"
    FAST = "fast"
    FASTV2 = "fastv2"
    INCREMENTAL = "incremental"
    INCREMENTAL3 = "incremental3"
    RECURSIVE = "recursive"
    PROPOSITIONAL = "propositional"
    TAIL_SIGNATURE = "tail-signature"
    BOUNDED_TREEWIDTH = "bounded-treewidth"

    def __str__(self) -> str:
        return self.value


class AlgoMaturity(Enum):
    """User-facing readiness of a registered algorithm."""

    STABLE = "stable"
    BETA = "beta"
    EXPERIMENTAL = "experimental"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AlgoOptions:
    """User-selectable strategy knobs for algorithm preparation and solving."""

    evidence_strategy: EvidenceStrategy | None = None
    existential_strategy: ExistentialStrategy | None = None
    linear_order_encoding: LinearOrderEncoding | None = None
    weight_options: WeightOptions = field(default_factory=WeightOptions)


@dataclass(frozen=True)
class AlgoSpec:
    """Small engine-facing contract for one algorithm."""

    name: AlgoName
    resolve_options: Callable[["FeatureSet", AlgoOptions | None], AlgoOptions]
    prepare: Callable[["Problem", AlgoOptions], tuple["PreparedBranch", ...]]
    solve: SolveFn
    maturity: AlgoMaturity = AlgoMaturity.STABLE
    external_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlgoInput:
    algo: AlgoName | None
    options: AlgoOptions
    arithmetic: ArithmeticContext

    def include_order_factorial(self) -> bool:
        """Whether result decoding should restore the linear-order factor."""

        return True


@dataclass(frozen=True)
class PreparedBranch:
    problem: "Problem | ReducedProblem"
    algo_input: AlgoInput
    decoder: Callable[..., object]


def option_resolver(
    *,
    algo: AlgoName,
    default_unary_evidence: EvidenceStrategy | DefaultEvidenceStrategy,
    supported_unary_evidence: tuple[EvidenceStrategy, ...] = (),
    default_existential_strategy: ExistentialStrategy = ExistentialStrategy.SKOLEM,
    supported_existential_strategies: tuple[ExistentialStrategy, ...] = (
        ExistentialStrategy.SKOLEM,
    ),
    supports_linear_order: bool = False,
    supports_predk_or_circular: bool = False,
    supports_mod_counting: bool = False,
    supports_binary_evidence: bool = False,
) -> Callable[["FeatureSet", AlgoOptions | None], AlgoOptions]:
    from wfomc.fol.grounding import resolve_linear_order_encoding

    def resolve_options(
        features: "FeatureSet", options: AlgoOptions | None = None
    ) -> AlgoOptions:
        options = options if options is not None else AlgoOptions()
        resolved_linear_order_encoding = resolve_linear_order_encoding(
            options.linear_order_encoding
        )
        requested_evidence_strategy = options.evidence_strategy
        resolved_evidence_strategy = (
            requested_evidence_strategy
            if requested_evidence_strategy is not None
            else _resolve_unary_evidence_strategy(
                default_unary_evidence,
                features,
                resolved_linear_order_encoding,
            )
        )
        resolved = AlgoOptions(
            evidence_strategy=resolved_evidence_strategy,
            existential_strategy=(
                options.existential_strategy or default_existential_strategy
            ),
            linear_order_encoding=resolved_linear_order_encoding,
            weight_options=options.weight_options,
        )
        _validate_supported_features(
            algo=algo,
            features=features,
            options=resolved,
            supported_unary_evidence=supported_unary_evidence,
            supported_existential_strategies=supported_existential_strategies,
            supports_linear_order=supports_linear_order,
            supports_predk_or_circular=supports_predk_or_circular,
            supports_mod_counting=supports_mod_counting,
            supports_binary_evidence=supports_binary_evidence,
        )
        return resolved

    return resolve_options


# ---------------------------------------------------------------------------
# Shared logical/numeric preparation helpers
# ---------------------------------------------------------------------------


def reduce_unary_evidence_for_options(
    problem: "ReducedProblem", *, options: AlgoOptions
) -> "ReducedProblem | ProblemWithDecoder":
    strategy = options.evidence_strategy or EvidenceStrategy.NONE
    if strategy is EvidenceStrategy.LIFTED_PROFILES:
        from wfomc.reduction import reduce_unary_evidence_to_profile_capacity

        return reduce_unary_evidence_to_profile_capacity(problem)
    if strategy is EvidenceStrategy.CCS:
        from wfomc.reduction import reduce_unary_evidence_to_cardinality_constraints

        return reduce_unary_evidence_to_cardinality_constraints(problem)
    return problem


def _compile_reduced_weights(
    problem: "ReducedProblem",
    arithmetic: ArithmeticContext,
) -> dict[object, tuple[object, object]]:
    """Compile the reduced problem's raw weights into its arithmetic ring."""
    compiled = compile_weight_mapping(dict(problem.weights), arithmetic)
    return dict(sorted(compiled.items(), key=lambda item: str(item[0])))


def compile_source_arithmetic(
    problem: "Problem",
    options: AlgoOptions,
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


def compile_reduced_problem(
    problem: "ReducedProblem",
    options: AlgoOptions,
) -> tuple["CompiledProblem", "FeatureSet"]:
    """Compile one reduced branch into its quantifier-free numeric problem."""

    from wfomc.problem import CompiledProblem

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
        degree_limits=problem.internal_weight_degree_limits,
    )
    logger.info(
        "Prepared arithmetic: backend=%s solver_symbols=%d output_symbols=%d",
        backend,
        len(solver_symbols),
        len(output_symbols),
    )
    compiled = CompiledProblem(
        sentence=sentence,
        arithmetic=arithmetic,
        domain=problem.domain,
        weights=_compile_reduced_weights(problem, arithmetic),
        evidence=problem.evidence,
        profile_capacity_constraint=problem.profile_capacity_constraint,
        circular_order_size=problem.circular_order_size,
    )
    from wfomc.engine.features import analyze_features

    return compiled, analyze_features(compiled)


def _resolve_unary_evidence_strategy(
    strategy: EvidenceStrategy | DefaultEvidenceStrategy,
    features: "FeatureSet",
    linear_order_encoding: "LinearOrderEncoding",
) -> EvidenceStrategy:
    if not features.has_unary_evidence:
        return EvidenceStrategy.NONE
    if isinstance(strategy, EvidenceStrategy):
        return strategy
    return strategy(features, linear_order_encoding)


def _linear_order_encoding(
    encoding: LinearOrderEncoding | str | None,
) -> LinearOrderEncoding:
    from wfomc.fol.grounding import resolve_linear_order_encoding

    return resolve_linear_order_encoding(encoding)


def _validate_supported_features(
    *,
    algo: AlgoName,
    features: FeatureSet,
    options: AlgoOptions,
    supported_unary_evidence: SupportedEvidence,
    supported_existential_strategies: tuple[ExistentialStrategy, ...],
    supports_linear_order: bool,
    supports_predk_or_circular: bool,
    supports_mod_counting: bool,
    supports_binary_evidence: bool,
) -> None:
    evidence_strategy = options.evidence_strategy
    if evidence_strategy is None:
        evidence_strategy = EvidenceStrategy.NONE
    rejection_reason = _rejection_reason(
        algo=algo,
        features=features,
        evidence_strategy=evidence_strategy,
        supported_unary_evidence=supported_unary_evidence,
        existential_strategy=options.existential_strategy,
        supported_existential_strategies=supported_existential_strategies,
        supports_linear_order=supports_linear_order,
        supports_predk_or_circular=supports_predk_or_circular,
        supports_mod_counting=supports_mod_counting,
        supports_binary_evidence=supports_binary_evidence,
    )
    if rejection_reason is not None:
        raise UnsupportedFeatureError(rejection_reason)


def _rejection_reason(
    *,
    algo: AlgoName,
    features: "FeatureSet",
    evidence_strategy: EvidenceStrategy,
    supported_unary_evidence: SupportedEvidence,
    existential_strategy: ExistentialStrategy | None,
    supported_existential_strategies: tuple[ExistentialStrategy, ...],
    supports_linear_order: bool,
    supports_predk_or_circular: bool,
    supports_mod_counting: bool,
    supports_binary_evidence: bool,
) -> str | None:
    if existential_strategy not in supported_existential_strategies:
        selected = existential_strategy.value if existential_strategy else "none"
        return f"{algo.value} does not support {selected} existential strategy"
    if features.has_binary_evidence and not supports_binary_evidence:
        return (
            f"{algo.value} does not support ground binary evidence; "
            "binary evidence must not be silently ignored"
        )
    if features.has_linear_order and not supports_linear_order:
        return f"{algo.value} does not support linear order features"
    if (
        features.has_predk or features.has_circular_pred
    ) and not supports_predk_or_circular:
        return f"{algo.value} does not support PREDk/CircularPred features"
    if features.has_mod_counting and not supports_mod_counting:
        return f"{algo.value} does not support mod counting"
    if (
        features.has_unary_evidence
        and evidence_strategy is not EvidenceStrategy.NONE
        and evidence_strategy not in supported_unary_evidence
    ):
        return (
            f"{algo.value} does not support {evidence_strategy.value} "
            "unary evidence strategy"
        )
    return None


_SPEC_MODULES: dict[AlgoName, str] = {
    AlgoName.STANDARD: "wfomc.algo.standard.spec",
    AlgoName.FAST: "wfomc.algo.fast.spec",
    AlgoName.FASTV2: "wfomc.algo.fastv2.spec",
    AlgoName.INCREMENTAL: "wfomc.algo.incremental.spec",
    AlgoName.INCREMENTAL3: "wfomc.algo.incremental3.spec",
    AlgoName.RECURSIVE: "wfomc.algo.recursive.spec",
    AlgoName.PROPOSITIONAL: "wfomc.algo.propositional.spec",
    AlgoName.TAIL_SIGNATURE: "wfomc.algo.tail_signature.spec",
    AlgoName.BOUNDED_TREEWIDTH: "wfomc.algo.treewidth.spec",
}
_SPEC_CACHE: dict[AlgoName, AlgoSpec] = {}


def algo_spec(algo: AlgoName | str) -> AlgoSpec:
    selected = algo if isinstance(algo, AlgoName) else AlgoName(algo)
    cached = _SPEC_CACHE.get(selected)
    if cached is not None:
        return cached

    spec = import_module(_SPEC_MODULES[selected]).SPEC
    if not isinstance(spec, AlgoSpec):
        raise TypeError(f"{selected.value} spec module did not expose an AlgoSpec")
    if spec.name is not selected:
        raise ValueError(f"{selected.value} spec declared name {spec.name.value!r}")
    _SPEC_CACHE[selected] = spec
    return spec


__all__ = [
    "AlgoInput",
    "AlgoMaturity",
    "AlgoName",
    "AlgoOptions",
    "AlgoSpec",
    "EvidenceStrategy",
    "ExistentialStrategy",
    "PreparedBranch",
    "algo_spec",
    "option_resolver",
    "compile_source_arithmetic",
    "reduce_unary_evidence_for_options",
    "compile_reduced_problem",
]
