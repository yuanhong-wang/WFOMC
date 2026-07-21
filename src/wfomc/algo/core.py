"""Shared algorithm contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, TypeAlias

from wfomc.arithmetic import ArithmeticContext
from wfomc.errors import UnsupportedFeatureError
from wfomc.options import (
    BoundaryProfileOptions,
    EvidenceStrategy,
    ExistentialStrategy,
    WeightOptions,
)
from wfomc.stages import CompiledReducedBranch, GroundingProblem


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from wfomc.fol.grounding import LinearOrderEncoding
    from wfomc.problem import Domain
    from wfomc.stages import (
        CompiledBranchInstance,
        FeatureSet,
    )
    from wfomc.result import WFOMCResult


SupportedEvidence = tuple[EvidenceStrategy, ...]
AlgoBranch: TypeAlias = CompiledReducedBranch | GroundingProblem


class DefaultEvidenceStrategy(Protocol):
    """Choose the default evidence strategy from analyzed source features."""

    def __call__(
        self,
        features: "FeatureSet",
        linear_order_encoding: "LinearOrderEncoding",
    ) -> EvidenceStrategy: ...


class SolveFn(Protocol):
    """Protocol for ``spec.solve``: ``AlgoInput × SolveContext → WFOMCResult``."""

    def __call__(
        self,
        algo_input: "AlgoInput",
        context: "SolveContext | None",
    ) -> "WFOMCResult": ...


class AlgoName(Enum):
    """Stable public names for WFOMC algorithms."""

    # Baseline lifted FO2 WFOMC algorithm.
    STANDARD = "standard"
    # Cell-graph fast WFOMC algorithm.
    FAST = "fast"
    # Optimized fast algorithm with modified cell symmetry.
    FASTV2 = "fastv2"
    # Incremental lifted algorithm for ordered structures.
    INCREMENTAL = "incremental"
    # Incremental algorithm with native counting-quantifier state.
    INCREMENTAL3 = "incremental3"
    # Recursive lifted algorithm for ordered structures.
    RECURSIVE = "recursive"
    # Reduction-independent source grounding followed by Ganak counting.
    PROPOSITIONAL = "propositional"
    # Logical reduction followed by quantifier-free grounding and Ganak.
    PROPOSITIONAL_REDUCED = "propositional-reduced"
    # Native binary-decomposition DP over cell boundary profiles.
    BOUNDARY_PROFILE = "boundary-profile"
    # Extension point for a future bounded-treewidth solver.
    BOUNDED_TREEWIDTH = "bounded-treewidth"

    def __str__(self) -> str:
        return self.value


class AlgoMaturity(Enum):
    """User-facing readiness of a registered algorithm."""

    # Supported for normal production use and exposed by the CLI.
    STABLE = "stable"
    # Runnable and CLI-visible, but still subject to documented limitations.
    BETA = "beta"
    # Registered only as an extension point and not currently runnable.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AlgoOptions:
    """User-selectable strategy knobs for algorithm preparation and solving."""

    # Unary-evidence preparation strategy, or None for the algorithm default.
    evidence_strategy: EvidenceStrategy | None = None
    # Incremental3-only existential reduction; None selects its counting default.
    existential_strategy: ExistentialStrategy | None = None
    # Propositional order encoding; None lets the resolver choose a sound default.
    linear_order_encoding: LinearOrderEncoding | None = None
    # Numeric precision and symbolic-arithmetic backend selection.
    weight_options: WeightOptions = field(default_factory=WeightOptions)
    # Boundary-Profile-only tree planning controls.
    boundary_profile_options: BoundaryProfileOptions = field(
        default_factory=BoundaryProfileOptions
    )


@dataclass(frozen=True)
class SolveContext:
    """Algorithm-visible external dependencies for one solve call."""

    # Explicit Ganak executable for propositional solving.
    ganak_path: str | None = None


class ReducedInputTemplate(ABC):
    """Nominal contract for templates instantiated from a reduced branch."""

    @abstractmethod
    def instantiate(
        self,
        concrete: "CompiledBranchInstance",
    ) -> "AlgoInput":
        """Create one concrete algorithm input."""


class GroundingInputTemplate(ABC):
    """Nominal contract for templates instantiated directly from a domain."""

    @abstractmethod
    def instantiate(
        self,
        domain: "Domain",
    ) -> "AlgoInput":
        """Create one concrete grounding input."""


AlgoInputTemplate: TypeAlias = ReducedInputTemplate | GroundingInputTemplate


@dataclass(frozen=True)
class AlgoSpec:
    """Small engine-facing contract for one algorithm."""

    # Public registry key used by the Python API and CLI.
    name: AlgoName
    # Resolve defaults and reject unsupported source features or option values.
    resolve_options: Callable[["FeatureSet", AlgoOptions | None], AlgoOptions]
    # Evaluate one algorithm-owned input and return its undecoded branch result.
    solve: SolveFn
    # Build one reusable algorithm-owned input template.
    build_input_template: Callable[
        [AlgoBranch, Hashable, AlgoOptions],
        AlgoInputTemplate,
    ]
    # Select a small structural key for the input-template cache.
    input_template_key: Callable[
        [AlgoBranch, "Domain"],
        Hashable,
    ] | None = None
    # Whether the engine applies logical reductions before numeric compilation.
    uses_reduction: bool = True
    # Whether the engine lowers counting quantifiers during reduction.
    reduce_counting_quantifiers: bool = True
    # Readiness level controlling whether the algorithm appears in the CLI.
    maturity: AlgoMaturity = AlgoMaturity.STABLE


@dataclass(frozen=True)
class AlgoInput:
    """Base fields shared by every materialized algorithm input."""

    # Branch-local numeric domain used for weights, solving, and decoding.
    arithmetic: ArithmeticContext

    def include_order_factorial(self) -> bool:
        """Whether result decoding should restore the linear-order factor."""

        return True


def option_resolver(
    *,
    algo: AlgoName,
    default_unary_evidence: EvidenceStrategy | DefaultEvidenceStrategy,
    supported_unary_evidence: tuple[EvidenceStrategy, ...] = (),
    default_existential_strategy: ExistentialStrategy | None = None,
    supported_existential_strategies: tuple[ExistentialStrategy | None, ...] = (None,),
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
        if (
            algo is not AlgoName.BOUNDARY_PROFILE
            and options.boundary_profile_options != BoundaryProfileOptions()
        ):
            raise UnsupportedFeatureError(
                "boundary-profile planning options are only configurable for "
                "boundary-profile"
            )
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
                options.existential_strategy
                if options.existential_strategy is not None
                else default_existential_strategy
            ),
            linear_order_encoding=resolved_linear_order_encoding,
            weight_options=options.weight_options,
            boundary_profile_options=options.boundary_profile_options,
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


def _validate_supported_features(
    *,
    algo: AlgoName,
    features: FeatureSet,
    options: AlgoOptions,
    supported_unary_evidence: SupportedEvidence,
    supported_existential_strategies: tuple[ExistentialStrategy | None, ...],
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
    supported_existential_strategies: tuple[ExistentialStrategy | None, ...],
    supports_linear_order: bool,
    supports_predk_or_circular: bool,
    supports_mod_counting: bool,
    supports_binary_evidence: bool,
) -> str | None:
    if existential_strategy not in supported_existential_strategies:
        selected = existential_strategy.value if existential_strategy else "none"
        return (
            f"existential strategy is only configurable for incremental3; "
            f"got {selected!r} for {algo.value}"
        )
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
    AlgoName.PROPOSITIONAL_REDUCED: "wfomc.algo.propositional.reduced_spec",
    AlgoName.BOUNDARY_PROFILE: "wfomc.algo.boundary_profile.spec",
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
    "AlgoInputTemplate",
    "AlgoBranch",
    "AlgoMaturity",
    "AlgoName",
    "AlgoOptions",
    "AlgoSpec",
    "EvidenceStrategy",
    "ExistentialStrategy",
    "GroundingInputTemplate",
    "ReducedInputTemplate",
    "SolveContext",
    "algo_spec",
    "option_resolver",
]
