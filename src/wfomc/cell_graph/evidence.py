"""Materialize reduced unary-evidence profiles against concrete cells."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from functools import reduce
from typing import TYPE_CHECKING, Iterator, Protocol

from wfomc.evidence.profile import ProfileCapacityConstraint
from wfomc.fol import Formula, Literal, Predicate, true
from wfomc.multinomial import MultinomialCoefficients, multinomial

if TYPE_CHECKING:
    from wfomc.arithmetic import ArithmeticContext, ArithmeticValue


class CellConfigCoefficientBasis(Enum):
    ABSOLUTE = "absolute"
    RELATIVE_TO_CELL_MULTINOMIAL = "relative-to-cell-multinomial"


class _CellEvidenceView(Protocol):
    def is_positive(self, predicate: Predicate) -> bool: ...


@dataclass(frozen=True)
class CellEvidenceAllocation:
    """Compatibility and counting data for profiles assigned to cells."""

    evidence_profile_sizes: tuple[int, ...]
    compatible_evidence_profiles_by_cell: tuple[tuple[int, ...], ...]
    compatible_cells_by_evidence_profile: tuple[tuple[int, ...], ...]
    evidence_assignment_count: Fraction = Fraction(1)
    profile_literals: tuple[frozenset[Literal], ...] = ()

    @classmethod
    def unconstrained(
        cls,
        n_cells: int,
        domain_size: int,
    ) -> "CellEvidenceAllocation":
        all_cells = tuple(range(n_cells))
        return cls(
            evidence_profile_sizes=(domain_size,),
            compatible_evidence_profiles_by_cell=tuple((0,) for _ in range(n_cells)),
            compatible_cells_by_evidence_profile=(all_cells,),
        )

    @classmethod
    def from_constraint(
        cls,
        constraint: ProfileCapacityConstraint,
        cells: tuple[_CellEvidenceView, ...],
    ) -> "CellEvidenceAllocation":
        compatible_profiles_by_cell = tuple(
            tuple(
                profile_idx
                for profile_idx, profile in enumerate(constraint.profiles)
                if all(
                    cell.is_positive(literal.predicate) == literal.positive
                    for literal in profile.literals
                )
            )
            for cell in cells
        )
        compatible_cells_by_profile = tuple(
            tuple(
                cell_idx
                for cell_idx, profile_indices in enumerate(compatible_profiles_by_cell)
                if profile_idx in profile_indices
            )
            for profile_idx in range(len(constraint.profiles))
        )
        return cls(
            evidence_profile_sizes=tuple(
                profile.size for profile in constraint.profiles
            ),
            compatible_evidence_profiles_by_cell=compatible_profiles_by_cell,
            compatible_cells_by_evidence_profile=compatible_cells_by_profile,
            evidence_assignment_count=constraint.assignment_count,
            profile_literals=tuple(profile.literals for profile in constraint.profiles),
        )

    @property
    def compatible_cell_indices(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, profile_indices in enumerate(
                self.compatible_evidence_profiles_by_cell
            )
            if profile_indices
        )

    @property
    def compatibility_pair_count(self) -> int:
        return sum(
            len(profile_indices)
            for profile_indices in self.compatible_evidence_profiles_by_cell
        )

    def iter_config_coefficients(
        self,
        arithmetic: "ArithmeticContext",
        basis: CellConfigCoefficientBasis = CellConfigCoefficientBasis.ABSOLUTE,
    ) -> Iterator[tuple[tuple[int, ...], "ArithmeticValue"]]:
        n_cells = len(self.compatible_evidence_profiles_by_cell)
        states = {(0,) * n_cells: arithmetic.one()}
        for profile_idx, size in enumerate(self.evidence_profile_sizes):
            compatible = self.compatible_cells_by_evidence_profile[profile_idx]
            if not compatible:
                if size == 0:
                    continue
                return

            new_states = {}
            for base_config, coefficient in states.items():
                for distribution in multinomial(len(compatible), size):
                    new_config = list(base_config)
                    for cell_idx, count in zip(compatible, distribution):
                        new_config[cell_idx] += count
                    config = tuple(new_config)
                    term = arithmetic.multiply(
                        coefficient,
                        arithmetic.from_int(
                            MultinomialCoefficients.coef(distribution)
                        ),
                    )
                    new_states[config] = arithmetic.add(
                        new_states.get(config, arithmetic.zero()),
                        term,
                    )
            states = new_states

        for config, coefficient in states.items():
            if basis is CellConfigCoefficientBasis.RELATIVE_TO_CELL_MULTINOMIAL:
                coefficient = arithmetic.multiply(
                    coefficient,
                    arithmetic.from_fraction(
                        1,
                        MultinomialCoefficients.coef(config),
                    ),
                )
            yield config, coefficient

    def initial_remaining_counts(self) -> tuple[int, ...]:
        return self.evidence_profile_sizes

    def next_remaining_counts(
        self,
        remaining_counts: tuple[int, ...],
        cell_index: int,
    ) -> tuple[tuple[int, ...], ...]:
        transitions = []
        for profile_idx in self.compatible_evidence_profiles_by_cell[cell_index]:
            if remaining_counts[profile_idx] <= 0:
                continue
            transitions.append(
                tuple(
                    count - 1 if idx == profile_idx else count
                    for idx, count in enumerate(remaining_counts)
                )
            )
        return tuple(transitions)

    def normalize_ordered_evidence_assignments(
        self,
        value: "ArithmeticValue",
        arithmetic: "ArithmeticContext",
    ) -> "ArithmeticValue":
        return value / arithmetic.coerce(self.evidence_assignment_count)


def materialize_cell_evidence(
    constraint: ProfileCapacityConstraint | None,
    cells: tuple[_CellEvidenceView, ...],
) -> CellEvidenceAllocation | None:
    """Bind one reduced profile constraint to a concrete cell set."""

    if constraint is None or constraint.is_empty:
        return None
    return CellEvidenceAllocation.from_constraint(constraint, cells)


def required_profile_predicates(
    constraint: ProfileCapacityConstraint | None,
) -> frozenset[Predicate]:
    """Predicates that must remain visible while enumerating cells."""

    if constraint is None or constraint.is_empty:
        return frozenset()
    return frozenset(
        literal.predicate
        for profile in constraint.profiles
        for literal in profile.literals
    )


def profile_cell_formulas(
    constraint: ProfileCapacityConstraint | None,
) -> tuple[Formula, ...] | None:
    """Restrict cell enumeration when every profile fixes unary literals."""

    if constraint is None or any(
        not profile.literals for profile in constraint.profiles
    ):
        return None
    return tuple(
        reduce(
            lambda left, right: left & right,
            (
                literal.atom if literal.positive else ~literal.atom
                for literal in profile.literals
            ),
            true(),
        )
        for profile in constraint.profiles
    )


__all__ = [
    "CellConfigCoefficientBasis",
    "CellEvidenceAllocation",
    "materialize_cell_evidence",
    "profile_cell_formulas",
    "required_profile_predicates",
]
