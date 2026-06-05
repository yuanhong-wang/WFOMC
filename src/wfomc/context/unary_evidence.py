"""Prepare unary evidence and compile it against concrete cell graphs.

``UnaryEvidencePlan`` owns strategy-specific preparation. Its
``UnaryEvidencePartition`` groups the domain into fixed-size ``EvidenceProfile``
objects, and each cell graph compiles that partition into a
``CellEvidenceAllocation`` for algorithm-specific elimination.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Iterable, Iterator, TYPE_CHECKING

from wfomc.fol import (
    AUXILIARY_PRED_NAME,
    AtomicFormula,
    Const,
    Pred,
    QFFormula,
    X,
    bot,
    exactly_one_qf,
    new_predicate,
    top,
)
from flint import fmpq as Rational

from wfomc.utils import (
    MultinomialCoefficients,
    RingElement,
    multinomial,
)

if TYPE_CHECKING:
    from wfomc.cell_graph import Cell
    from wfomc.fol.sc2 import SC2


# ---------------------------------------------------------------------------
# Public strategy and algorithm-facing coefficient options
# ---------------------------------------------------------------------------

class UnaryEvidenceStrategy(Enum):
    """Public selector for unary evidence handling."""

    AUTO = "auto"
    CCS = "ccs"

    def __str__(self) -> str:
        return self.value


class CellConfigCoefficientBasis(Enum):
    """Basis used by configuration-enumerating algorithms."""

    ABSOLUTE = "absolute"
    RELATIVE_TO_CELL_MULTINOMIAL = "relative-to-cell-multinomial"


# ---------------------------------------------------------------------------
# Evidence-induced profiles
# ---------------------------------------------------------------------------

def _literal_key(literal: AtomicFormula) -> tuple:
    return (
        literal.pred.name,
        literal.pred.arity,
        literal.positive,
        tuple(str(arg) for arg in literal.args),
    )


def _profile_key(literals: frozenset[AtomicFormula]) -> tuple:
    return tuple(sorted(_literal_key(literal) for literal in literals))


@dataclass(frozen=True)
class EvidenceProfile:
    """A fixed-size partial unary assignment induced by evidence."""

    evidence: frozenset[AtomicFormula]
    size: int

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError("Evidence profile sizes must be positive.")
        for literal in self.evidence:
            if literal.pred.arity != 1 or literal.args != (X,):
                raise ValueError(
                    "Evidence profiles must contain unary literals over X."
                )


@dataclass(frozen=True)
class UnaryEvidencePartition:
    """A deterministic domain partition induced by unary evidence."""

    evidence_profiles: tuple[EvidenceProfile, ...]
    domain_size: int

    def __post_init__(self):
        if self.domain_size < 0:
            raise ValueError("Domain size must be non-negative.")
        if (
            sum(evidence_profile.size for evidence_profile in self.evidence_profiles)
            != self.domain_size
        ):
            raise ValueError(
                "Evidence profile sizes must sum to the domain size."
            )

    @classmethod
    def from_evidence(
        cls,
        evidence: Iterable[AtomicFormula] | None,
        domain: set[Const],
    ) -> "UnaryEvidencePartition":
        domain_set = set(domain)
        element2evidence = organize_evidence(evidence or set())

        for element in element2evidence:
            if element not in domain_set:
                raise ValueError(
                    f"Evidence must be consistent with the domain: "
                    f"{element} not in {domain_set}."
                )

        sizes: dict[frozenset[AtomicFormula], int] = defaultdict(int)
        for element in sorted(domain_set, key=lambda const: const.name):
            profile = frozenset(element2evidence.get(element, set()))
            sizes[profile] += 1

        evidence_profiles = tuple(
            EvidenceProfile(profile, sizes[profile])
            for profile in sorted(sizes, key=_profile_key)
        )
        return cls(evidence_profiles, len(domain_set))

    @property
    def predicates(self) -> frozenset[Pred]:
        return frozenset(
            literal.pred
            for evidence_profile in self.evidence_profiles
            for literal in evidence_profile.evidence
        )

    @property
    def covers_all_elements(self) -> bool:
        return bool(self.evidence_profiles) and all(
            evidence_profile.evidence
            for evidence_profile in self.evidence_profiles
        )

    @property
    def evidence_assignment_count(self) -> Rational:
        """Number of fixed-size evidence-profile assignments over a labeled domain."""
        value = Rational(math.factorial(self.domain_size), 1)
        for evidence_profile in self.evidence_profiles:
            value /= Rational(math.factorial(evidence_profile.size), 1)
        return value

    def coverage_formula(self) -> QFFormula:
        """Prune cells that are incompatible with every evidence profile."""
        if any(not evidence_profile.evidence for evidence_profile in self.evidence_profiles):
            return top

        disjuncts = [
            reduce(lambda left, right: left & right, evidence_profile.evidence, top)
            for evidence_profile in self.evidence_profiles
        ]
        return reduce(lambda left, right: left | right, disjuncts, bot)

    def compile_for_cells(
        self,
        cells: tuple["Cell", ...],
    ) -> "CellEvidenceAllocation":
        """Compile evidence profiles against one concrete cell graph."""
        return CellEvidenceAllocation.from_partition(self, cells)


# ---------------------------------------------------------------------------
# Cell-graph-specific evidence allocation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellEvidenceAllocation:
    """Feasible allocations of evidence profiles to cells in one cell graph."""

    evidence_profile_sizes: tuple[int, ...]
    compatible_evidence_profiles_by_cell: tuple[tuple[int, ...], ...]
    compatible_cells_by_evidence_profile: tuple[tuple[int, ...], ...]
    evidence_assignment_count: Rational

    @classmethod
    def unconstrained(
        cls,
        n_cells: int,
        domain_size: int,
    ) -> "CellEvidenceAllocation":
        """Build the one-unrestricted-evidence-profile allocation."""
        all_cells = tuple(range(n_cells))
        return cls(
            evidence_profile_sizes=(domain_size,),
            compatible_evidence_profiles_by_cell=tuple((0,) for _ in range(n_cells)),
            compatible_cells_by_evidence_profile=(all_cells,),
            evidence_assignment_count=Rational(1, 1),
        )

    @classmethod
    def from_partition(
        cls,
        partition: UnaryEvidencePartition,
        cells: tuple["Cell", ...],
    ) -> "CellEvidenceAllocation":
        compatible_evidence_profiles_by_cell = tuple(
            tuple(
                evidence_profile_idx
                for evidence_profile_idx, evidence_profile in enumerate(
                    partition.evidence_profiles
                )
                if all(
                    cell.is_positive(literal.pred) == literal.positive
                    for literal in evidence_profile.evidence
                )
            )
            for cell in cells
        )
        compatible_cells_by_evidence_profile = tuple(
            tuple(
                cell_idx
                for cell_idx, evidence_profile_indices in enumerate(
                    compatible_evidence_profiles_by_cell
                )
                if evidence_profile_idx in evidence_profile_indices
            )
            for evidence_profile_idx in range(len(partition.evidence_profiles))
        )
        return cls(
            evidence_profile_sizes=tuple(
                evidence_profile.size for evidence_profile in partition.evidence_profiles
            ),
            compatible_evidence_profiles_by_cell=compatible_evidence_profiles_by_cell,
            compatible_cells_by_evidence_profile=compatible_cells_by_evidence_profile,
            evidence_assignment_count=partition.evidence_assignment_count,
        )

    @property
    def compatible_cell_indices(self) -> tuple[int, ...]:
        return tuple(
            idx
            for idx, evidence_profiles in enumerate(
                self.compatible_evidence_profiles_by_cell
            )
            if evidence_profiles
        )

    @property
    def compatibility_pair_count(self) -> int:
        return sum(
            len(evidence_profiles)
            for evidence_profiles in self.compatible_evidence_profiles_by_cell
        )

    def iter_config_coefficients(
        self,
        basis: CellConfigCoefficientBasis = (
            CellConfigCoefficientBasis.ABSOLUTE
        ),
    ) -> Iterator[tuple[tuple[int, ...], RingElement]]:
        """Yield cell-count configurations and evidence-consistent multiplicities."""
        n_cells = len(self.compatible_evidence_profiles_by_cell)
        states: dict[tuple[int, ...], RingElement] = {
            (0,) * n_cells: Rational(1, 1)
        }

        for evidence_profile_idx, size in enumerate(self.evidence_profile_sizes):
            compatible = self.compatible_cells_by_evidence_profile[evidence_profile_idx]
            if not compatible:
                return

            new_states: dict[tuple[int, ...], RingElement] = defaultdict(
                lambda: Rational(0, 1)
            )
            for base_config, acc in states.items():
                for dist in multinomial(len(compatible), size):
                    new_config = list(base_config)
                    for cell_idx, count in zip(compatible, dist):
                        new_config[cell_idx] += count
                    new_states[tuple(new_config)] += (
                        acc * MultinomialCoefficients.coef(dist)
                    )
            states = new_states

        for config, coefficient in states.items():
            if basis == (
                CellConfigCoefficientBasis.RELATIVE_TO_CELL_MULTINOMIAL
            ):
                coefficient *= Rational(
                    1, MultinomialCoefficients.coef(config)
                )
            yield config, coefficient

    def initial_remaining_counts(self) -> tuple[int, ...]:
        return self.evidence_profile_sizes

    def next_remaining_counts(
        self,
        remaining_counts: tuple[int, ...],
        cell_index: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Assign one element to each compatible non-empty evidence profile."""
        transitions = []
        for evidence_profile_idx in self.compatible_evidence_profiles_by_cell[cell_index]:
            if remaining_counts[evidence_profile_idx] <= 0:
                continue
            transitions.append(
                tuple(
                    count - 1 if idx == evidence_profile_idx else count
                    for idx, count in enumerate(remaining_counts)
                )
            )
        return tuple(transitions)

    def normalize_ordered_evidence_assignments(
        self,
        value: RingElement,
    ) -> RingElement:
        """Remove evidence-profile sequences counted by threaded placement paths."""
        return value / self.evidence_assignment_count


# ---------------------------------------------------------------------------
# Context preparation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnaryEvidencePlan:
    """Strategy-specific inputs ready to be applied by a WFOMC context."""

    strategy: UnaryEvidenceStrategy
    partition: UnaryEvidencePartition
    formula: QFFormula
    cardinality_constraints: tuple[tuple[Pred, str, int], ...]
    repeat_factor: int

    @classmethod
    def build(
        cls,
        evidence: Iterable[AtomicFormula],
        domain: set[Const],
        sentence: "SC2",
        strategy: UnaryEvidenceStrategy,
    ) -> "UnaryEvidencePlan":
        _validate_exchangeability(sentence)
        partition = UnaryEvidencePartition.from_evidence(evidence, domain)

        if strategy == UnaryEvidenceStrategy.AUTO:
            return cls(
                strategy=strategy,
                partition=partition,
                formula=top,
                cardinality_constraints=(),
                repeat_factor=1,
            )
        if strategy == UnaryEvidenceStrategy.CCS:
            formula, constraints, repeat_factor = _encode_ccs(partition)
            return cls(
                strategy=strategy,
                partition=partition,
                formula=formula,
                cardinality_constraints=constraints,
                repeat_factor=repeat_factor,
            )
        raise ValueError(f"Unsupported unary evidence strategy: {strategy}")

    @property
    def uses_cell_allocation(self) -> bool:
        return self.strategy == UnaryEvidenceStrategy.AUTO

    @property
    def required_unary_preds(self) -> frozenset[Pred]:
        if not self.uses_cell_allocation:
            return frozenset()
        return self.partition.predicates

    def compile_for_cells(
        self,
        cells: list["Cell"] | tuple["Cell", ...],
    ) -> CellEvidenceAllocation | None:
        if not self.uses_cell_allocation:
            return None
        return self.partition.compile_for_cells(tuple(cells))


# ---------------------------------------------------------------------------
# Input normalization and private CCS implementation
# ---------------------------------------------------------------------------

def organize_evidence(
    evidence: Iterable[AtomicFormula],
) -> dict[Const, set[AtomicFormula]]:
    """Group ground unary evidence by named domain element."""
    element2evidence: dict[Const, set[AtomicFormula]] = defaultdict(set)
    for atom in evidence:
        if atom.pred.arity != 1 or len(atom.args) != 1:
            raise ValueError("Evidence must be unary.")
        element = atom.args[0]
        if not isinstance(element, Const):
            raise ValueError("Unary evidence must be ground.")
        normalized = atom.substitute({element: X})
        if ~normalized in element2evidence[element]:
            raise ValueError(
                f"Evidence must be consistent: {atom} and {~atom} both present."
            )
        element2evidence[element].add(normalized)
    return dict(element2evidence)


def _validate_exchangeability(sentence: "SC2") -> None:
    sentence_formulas = [
        sentence.uni_formula,
        *sentence.ext_formulas,
        *sentence.cnt_formulas,
    ]
    sentence_constants = set()
    for formula in sentence_formulas:
        if formula is not None:
            sentence_constants.update(formula.consts())
    if sentence_constants:
        raise ValueError(
            "Unary evidence grouping requires an exchangeable domain, but "
            "the sentence contains named constants: "
            f"{sorted(sentence_constants, key=lambda const: const.name)}"
        )


def _encode_ccs(
    partition: UnaryEvidencePartition,
) -> tuple[QFFormula, tuple[tuple[Pred, str, int], ...], int]:
    """Encode evidence profiles with auxiliary predicates and cardinalities."""
    formula = top
    aux_preds = []
    cardinality_constraints = []
    for evidence_profile in partition.evidence_profiles:
        aux_pred = new_predicate(1, AUXILIARY_PRED_NAME)
        aux_preds.append(aux_pred)
        aux_atom = aux_pred(X)
        if evidence_profile.evidence:
            formula &= aux_atom.implies(
                reduce(lambda left, right: left & right, evidence_profile.evidence)
            )
        cardinality_constraints.append((aux_pred, "=", evidence_profile.size))
    formula &= exactly_one_qf(aux_preds)
    return (
        formula,
        tuple(cardinality_constraints),
        int(partition.evidence_assignment_count),
    )


__all__ = [
    "CellConfigCoefficientBasis",
    "CellEvidenceAllocation",
    "EvidenceProfile",
    "UnaryEvidencePartition",
    "UnaryEvidencePlan",
    "UnaryEvidenceStrategy",
    "organize_evidence",
]
