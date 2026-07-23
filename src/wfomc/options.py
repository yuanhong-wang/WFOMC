"""Neutral strategy options shared by algorithms and reductions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class EvidenceStrategy(Enum):
    """Strategy for consuming unary evidence."""

    # No evidence encoding is needed because the source problem has no evidence.
    NONE = "none"
    # Lower evidence profiles to cardinality constraints before compilation.
    CCS = "ccs"
    # Preserve profile capacities for lifted configuration-coefficient counting.
    LIFTED_PROFILES = "lifted-profiles"
    # Emit one propositional unit clause for each ground evidence literal.
    GROUND_UNITS = "ground-units"
    # Materialize evidence as factors in a tree-decomposition backend.
    TREE_DECOMPOSITION_FACTORS = "tree-decomposition-factors"

    def __str__(self) -> str:
        return self.value


class ExistentialStrategy(Enum):
    """Strategy for existential normal-form sections."""

    # Convert existential sections to native count-at-least-one constraints.
    COUNTING = "counting"
    # Eliminate existential sections with exact weighted Skolemization.
    SKOLEM = "skolem"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class BoundaryProfileOptions:
    """Planning controls for the Boundary-Profile algorithm."""

    # Optional synthetic domain size used only to select one reusable BP tree.
    tree_reference_domain_size: int | None = None

    def __post_init__(self) -> None:
        if (
            self.tree_reference_domain_size is not None
            and self.tree_reference_domain_size < 0
        ):
            raise ValueError(
                "BoundaryProfileOptions.tree_reference_domain_size must be "
                "non-negative"
            )


@dataclass(frozen=True)
class WeightOptions:
    """User-facing weight-precision choices."""

    # Use exact rational arithmetic or rounded numerical arithmetic.
    precision: Literal["exact", "round"] = "exact"
    # Scalar/polynomial backend used when precision is rounded.
    rounded_backend: Literal["float", "arb"] = "arb"
    # Policy/backend used for exact symbolic weights and marker variables.
    exact_symbolic_backend: Literal["auto", "fmpq_mpoly", "fmpq_poly"] = "auto"

    def __post_init__(self) -> None:
        if self.precision not in {"exact", "round"}:
            raise ValueError(
                "WeightOptions.precision must be either 'exact' or 'round'"
            )
        if self.rounded_backend not in {"float", "arb"}:
            raise ValueError(
                "WeightOptions.rounded_backend must be either 'float' or 'arb'"
            )
        if self.exact_symbolic_backend not in {
            "auto",
            "fmpq_mpoly",
            "fmpq_poly",
        }:
            raise ValueError(
                "WeightOptions.exact_symbolic_backend must be 'auto', "
                "'fmpq_mpoly', or 'fmpq_poly'"
            )


__all__ = [
    "BoundaryProfileOptions",
    "EvidenceStrategy",
    "ExistentialStrategy",
    "WeightOptions",
]
