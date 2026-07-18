"""Domain-free reduced problems for staged algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import math
from typing import TYPE_CHECKING

from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.cardinality_constraints import (
    CardinalityConstraints,
    CardinalityTerm,
    Comparator,
    LinearCardinalityConstraint,
)
from wfomc.evidence import Evidence
from wfomc.evidence.data import UnaryEvidence
from wfomc.evidence.profile import EvidenceProfile, ProfileCapacityConstraint
from wfomc.fol import Literal
from wfomc.weights import (
    collect_output_weight_variables,
    compile_weight_mapping,
)

if TYPE_CHECKING:
    from wfomc.algo.core import AlgoOptions, EvidenceStrategy
    from wfomc.fol.normal_form import C2NormalForm
    from wfomc.problem import Domain, Problem


@dataclass(frozen=True)
class DomainExpr:
    """Small exact arithmetic expression evaluated from a domain size."""

    op: str
    args: tuple[object, ...] = ()

    @classmethod
    def constant(cls, value: int | Fraction) -> "DomainExpr":
        return cls("constant", (Fraction(value),))

    @classmethod
    def size(cls) -> "DomainExpr":
        return cls("size")

    def evaluate(self, domain_size: int) -> Fraction:
        values = tuple(
            arg.evaluate(domain_size) if isinstance(arg, DomainExpr) else arg
            for arg in self.args
        )
        if self.op == "constant":
            return Fraction(values[0])
        if self.op == "size":
            return Fraction(domain_size)
        if self.op == "add":
            return Fraction(values[0]) + Fraction(values[1])
        if self.op == "sub":
            return Fraction(values[0]) - Fraction(values[1])
        if self.op == "mul":
            return Fraction(values[0]) * Fraction(values[1])
        if self.op == "div":
            return Fraction(values[0]) / Fraction(values[1])
        if self.op == "floordiv":
            return Fraction(math.floor(Fraction(values[0]) / Fraction(values[1])))
        if self.op == "pow":
            exponent = _exact_int(Fraction(values[1]), "power exponent")
            return Fraction(values[0]) ** exponent
        if self.op == "factorial":
            value = _exact_int(Fraction(values[0]), "factorial argument")
            return Fraction(math.factorial(value))
        if self.op == "min":
            return min(Fraction(value) for value in values)
        raise ValueError(f"Unknown domain expression operation: {self.op!r}")

    def __add__(self, other: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("add", (self, _expr(other)))

    def __sub__(self, other: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("sub", (self, _expr(other)))

    def __mul__(self, other: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("mul", (self, _expr(other)))

    def __truediv__(self, other: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("div", (self, _expr(other)))

    def __floordiv__(self, other: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("floordiv", (self, _expr(other)))

    def power(self, exponent: int | "DomainExpr") -> "DomainExpr":
        return DomainExpr("pow", (self, _expr(exponent)))

    def factorial(self) -> "DomainExpr":
        return DomainExpr("factorial", (self,))

    @classmethod
    def minimum(cls, *values: "DomainExpr") -> "DomainExpr":
        if len(values) == 1:
            return values[0]
        return cls("min", tuple(values))


N = DomainExpr.size()


@dataclass(frozen=True)
class ReducedCardinalityConstraint:
    """Linear cardinality constraint with a domain-sized right-hand side."""

    terms: tuple[CardinalityTerm, ...]
    comparator: Comparator
    rhs: DomainExpr
    modulus: int | None = None

    def instantiate(self, domain_size: int) -> LinearCardinalityConstraint:
        return LinearCardinalityConstraint(
            terms=self.terms,
            comparator=self.comparator,
            rhs=_exact_int(self.rhs.evaluate(domain_size), "cardinality rhs"),
            modulus=self.modulus,
        )


@dataclass(frozen=True)
class ReducedProfileConstraint:
    """Unary-evidence profiles whose capacities may depend on the domain."""

    profiles: tuple[tuple[frozenset[Literal], DomainExpr], ...]

    def profile_sizes(self, domain_size: int) -> tuple[int, ...]:
        """Evaluate all capacities, including structurally absent zero profiles."""

        sizes = tuple(
            _exact_int(size.evaluate(domain_size), "evidence profile size")
            for _literals, size in self.profiles
        )
        if any(size < 0 for size in sizes):
            raise ValueError("Evidence profile sizes must be non-negative.")
        return sizes

    def active_profile_indices(self, domain_size: int) -> tuple[int, ...]:
        """Return the profiles present in one concrete domain."""

        return tuple(
            index
            for index, size in enumerate(self.profile_sizes(domain_size))
            if size > 0
        )

    def structural_constraint(
        self,
        active_profile_indices: tuple[int, ...] | None = None,
    ) -> ProfileCapacityConstraint:
        """Return a size-insensitive constraint for cell-graph construction."""

        indices = (
            tuple(range(len(self.profiles)))
            if active_profile_indices is None
            else active_profile_indices
        )
        return ProfileCapacityConstraint(
            profiles=tuple(
                EvidenceProfile(self.profiles[index][0], 1) for index in indices
            ),
            domain_size=len(indices),
        )

    def instantiate(self, domain_size: int) -> ProfileCapacityConstraint:
        sizes = self.profile_sizes(domain_size)
        profiles = tuple(
            EvidenceProfile(literals, size)
            for (literals, _expression), size in zip(self.profiles, sizes)
            if size > 0
        )
        assignment_count = Fraction(math.factorial(domain_size), 1)
        for profile in profiles:
            assignment_count /= math.factorial(profile.size)
        return ProfileCapacityConstraint(
            profiles=profiles,
            domain_size=domain_size,
            assignment_count=assignment_count,
        )

    def assignment_count_expr(self) -> DomainExpr:
        denominator = DomainExpr.constant(1)
        for _literals, size in self.profiles:
            denominator = denominator * size.factorial()
        return N.factorial() / denominator


@dataclass(frozen=True)
class DivideDecoderSpec:
    coefficient: DomainExpr


@dataclass(frozen=True)
class CardinalityDecoderSpec:
    constraints: tuple[ReducedCardinalityConstraint, ...]
    predicate_markers: tuple[tuple[object, str], ...]


DecoderStep = DivideDecoderSpec | CardinalityDecoderSpec


@dataclass(frozen=True)
class DecoderSpec:
    """Data-only composition of reduction corrections."""

    steps: tuple[DecoderStep, ...] = ()

    def append(self, step: DecoderStep) -> "DecoderSpec":
        return DecoderSpec(self.steps + (step,))

    def instantiate(self, domain_size: int):
        concrete_steps = []
        for step in self.steps:
            if isinstance(step, DivideDecoderSpec):
                concrete_steps.append(
                    ("divide", step.coefficient.evaluate(domain_size))
                )
            else:
                concrete_steps.append(
                    (
                        "cardinality",
                        CardinalityConstraints(
                            tuple(
                                constraint.instantiate(domain_size)
                                for constraint in step.constraints
                            )
                        ),
                        step.predicate_markers,
                    )
                )

        def decode(result: object, **kwargs: object) -> object:
            arithmetic = kwargs.get("arithmetic")
            if not isinstance(arithmetic, ArithmeticContext):
                raise TypeError("reduced-problem decoder requires ArithmeticContext")
            value = result
            for concrete in reversed(concrete_steps):
                if concrete[0] == "divide":
                    coefficient = Fraction(concrete[1])
                    value = arithmetic.multiply(
                        value,
                        arithmetic.from_fraction(
                            coefficient.denominator,
                            coefficient.numerator,
                        ),
                    )
                else:
                    from wfomc.reduction.reduce_cardinality_constraints import (
                        decode_cardinality_result,
                    )

                    value = decode_cardinality_result(
                        value,
                        concrete[1],
                        concrete[2],
                        arithmetic,
                    )
            return value

        return decode


@dataclass(frozen=True)
class ReducedProblem:
    """Domain-free logical branch produced by staged reductions."""

    normal_form: "C2NormalForm"
    weights: dict[object, tuple[object, object]] = field(default_factory=dict)
    cardinality_constraints: tuple[ReducedCardinalityConstraint, ...] = ()
    evidence: Evidence = field(default_factory=Evidence)
    profile_constraint: ReducedProfileConstraint | None = None
    # Unary CCS markers; unmarked elements form the empty evidence profile.
    ccs_profile_markers: tuple[object, ...] = ()
    ccs_unmarked_size: DomainExpr | None = None
    internal_weight_symbols: tuple[str, ...] = ()
    internal_weight_degree_limits: tuple[tuple[str, DomainExpr], ...] = ()
    decoder_spec: DecoderSpec = DecoderSpec()
    required_constants: frozenset[object] = frozenset()
    min_domain_size: int = 0
    max_domain_size: int | None = None

    @property
    def has_unary_evidence(self) -> bool:
        return not self.evidence.unary.is_empty

    @property
    def has_binary_evidence(self) -> bool:
        return not self.evidence.binary.is_empty

    @property
    def has_cardinality_constraints(self) -> bool:
        return bool(self.cardinality_constraints)

    @property
    def has_profile_capacity_constraint(self) -> bool:
        return self.profile_constraint is not None

    def applies(self, domain: "Domain") -> bool:
        return (
            self.required_constants <= domain.elements
            and domain.size >= self.min_domain_size
            and (self.max_domain_size is None or domain.size <= self.max_domain_size)
        )


def reduce_problem(
    problem: "Problem",
    options: "AlgoOptions",
    *,
    reduce_counting_quantifiers: bool = True,
) -> tuple[ReducedProblem, ...]:
    """Apply a declared reduction sequence without a concrete domain."""

    from wfomc.fol.normal_form import normalize, validate_normal_form

    normal_form = normalize(
        problem.sentence,
        reserved_predicate_names=problem.declared_predicate_names(),
    )
    validate_normal_form(normal_form)
    required = problem.required_domain_constants()
    minimum = max(
        len(required),
        1 if normal_form.requires_nonempty_domain else 0,
    )
    source_constraints = tuple(
        _reduced_constraint(constraint)
        for constraint in problem.cardinality_constraints.constraints
    )
    initial = ReducedProblem(
        normal_form=normal_form,
        weights=dict(problem.weights),
        cardinality_constraints=source_constraints,
        evidence=problem.evidence,
        required_constants=required,
        min_domain_size=minimum,
    )
    branches = _reduce_unary_evidence(
        initial,
        options.evidence_strategy,
    )
    result = []
    for branch in branches:
        counted = (
            _reduce_counting(branch)
            if reduce_counting_quantifiers
            else branch
        )
        from wfomc.reduction.reduce_existential_quantifiers import (
            reduce_existential_quantifiers,
        )

        existential = reduce_existential_quantifiers(
            counted,
            options=options,
        )
        result.append(_reduce_cardinality(existential))
    return tuple(result)


def _reduce_unary_evidence(
    problem: ReducedProblem,
    strategy: "EvidenceStrategy | None",
) -> tuple[ReducedProblem, ...]:
    from wfomc.algo.core import EvidenceStrategy

    unary = problem.evidence.unary
    if unary.is_empty or strategy in (None, EvidenceStrategy.NONE):
        return (problem,)
    if strategy is EvidenceStrategy.GROUND_UNITS:
        return (problem,)
    nonempty_profiles = _observed_profiles(
        unary,
        problem.required_constants,
    )
    observed_nonempty = sum(size for _literals, size in nonempty_profiles)
    profile = ReducedProfileConstraint(
        tuple(
            (literals, DomainExpr.constant(size))
            for literals, size in nonempty_profiles
        )
        + ((frozenset(), N - observed_nonempty),)
    )
    return (
        _apply_profile_strategy(
            problem,
            profile,
            strategy,
        ),
    )


def _observed_profiles(
    evidence: UnaryEvidence,
    required_constants: frozenset[object],
) -> tuple[tuple[frozenset[Literal], int], ...]:
    element_to_literals: dict[object, set[Literal]] = {}
    for literal in evidence.literals:
        ground = literal.to_ground_literal()
        constant = ground.terms[0]
        normalized = literal.to_profile_literal()
        element_literals = element_to_literals.setdefault(constant, set())
        if ~normalized in element_literals:
            raise ValueError(
                f"Evidence must be consistent for {constant}: {literal.predicate}"
            )
        element_literals.add(normalized)
    if not set(element_to_literals) <= set(required_constants):
        raise RuntimeError("evidence constants were not included in required constants")
    sizes: dict[frozenset[Literal], int] = {}
    for literals in element_to_literals.values():
        profile = frozenset(literals)
        sizes[profile] = sizes.get(profile, 0) + 1
    return tuple(
        (profile, sizes[profile]) for profile in sorted(sizes, key=_profile_sort_key)
    )


def _apply_profile_strategy(
    problem: ReducedProblem,
    profile: ReducedProfileConstraint,
    strategy: "EvidenceStrategy",
) -> ReducedProblem:
    from wfomc.algo.core import EvidenceStrategy

    cleared_evidence = replace(problem.evidence, unary=UnaryEvidence())
    if strategy is EvidenceStrategy.LIFTED_PROFILES:
        return replace(
            problem,
            profile_constraint=profile,
            evidence=cleared_evidence,
        )
    if strategy is not EvidenceStrategy.CCS:
        raise ValueError(f"Unsupported staged evidence strategy: {strategy}")

    from wfomc.fol import conjunction, true
    from wfomc.reduction.reduce_unary_evidence import (
        build_ccs_formula_and_constraints,
    )

    encoded_profiles = tuple(item for item in profile.profiles if item[0])
    structural = ReducedProfileConstraint(encoded_profiles).structural_constraint()
    formula_patch, simple_constraints = build_ccs_formula_and_constraints(
        structural,
        cover_domain=False,
    )
    qf_formula = problem.normal_form.qf_formula
    if qf_formula is None:
        qf_formula = true()
    profile_constraints = tuple(
        ReducedCardinalityConstraint(
            terms=(CardinalityTerm(predicate),),
            comparator=Comparator(comparator),
            rhs=size,
        )
        for (predicate, comparator, _rhs), (_literals, size) in zip(
            simple_constraints,
            encoded_profiles,
        )
    )
    unmarked_sizes = tuple(size for literals, size in profile.profiles if not literals)
    if len(unmarked_sizes) != 1:
        raise RuntimeError("Unary CCS expects exactly one unmarked profile")
    return replace(
        problem,
        normal_form=replace(
            problem.normal_form,
            qf_formula=conjunction(qf_formula, formula_patch),
        ),
        cardinality_constraints=(problem.cardinality_constraints + profile_constraints),
        evidence=cleared_evidence,
        ccs_profile_markers=tuple(
            predicate for predicate, _comparator, _rhs in simple_constraints
        ),
        ccs_unmarked_size=unmarked_sizes[0],
        decoder_spec=problem.decoder_spec.append(
            DivideDecoderSpec(profile.assignment_count_expr())
        ),
    )


def _reduce_counting(
    problem: ReducedProblem,
) -> ReducedProblem:
    normal_form = problem.normal_form
    if not normal_form.has_counting:
        return problem
    from wfomc.fol import conjunction, true
    from wfomc.reduction.reduce_counting_quantifiers import reduce_counting

    reduction = reduce_counting(
        normal_form,
        domain_size=1,
        rational_cls=Fraction,
        reserved_predicate_names=(str(predicate) for predicate in problem.weights),
    )
    generated = []
    global_count = len(normal_form.counts)
    for index, constraint in enumerate(reduction.cardinality_constraints.constraints):
        rhs = (
            DomainExpr.constant(constraint.rhs)
            if index < global_count
            else N * int(normal_form.forall_counts[index - global_count].count)
        )
        generated.append(
            ReducedCardinalityConstraint(
                terms=constraint.terms,
                comparator=constraint.comparator,
                rhs=rhs,
                modulus=constraint.modulus,
            )
        )
    repeat = DomainExpr.constant(1)
    for row_count in normal_form.forall_counts:
        repeat = repeat * DomainExpr.constant(
            math.factorial(int(row_count.count))
        ).power(N)
    weights = dict(problem.weights)
    weights.update(reduction.weight_map())
    qf_formula = normal_form.qf_formula
    if qf_formula is None:
        qf_formula = true()
    return replace(
        problem,
        normal_form=replace(
            normal_form,
            qf_formula=conjunction(qf_formula, reduction.formula_patch),
            counts=(),
            forall_counts=(),
            count_definitions=(),
        ),
        weights=weights,
        cardinality_constraints=(problem.cardinality_constraints + tuple(generated)),
        decoder_spec=problem.decoder_spec.append(DivideDecoderSpec(repeat)),
    )


def _reduce_cardinality(
    problem: ReducedProblem,
) -> ReducedProblem:
    constraints = problem.cardinality_constraints
    if not constraints:
        return problem
    predicates = sorted(
        {term.predicate for constraint in constraints for term in constraint.terms},
        key=str,
    )
    user_symbols = collect_output_weight_variables(problem)
    used_symbols = set(user_symbols) | set(problem.internal_weight_symbols)
    predicate_markers = []
    for index, predicate in enumerate(predicates):
        marker = f"__wfomc_cardinality_{index}"
        while marker in used_symbols:
            marker += "_"
        used_symbols.add(marker)
        predicate_markers.append((predicate, marker))
    predicate_markers_tuple = tuple(predicate_markers)
    marker_names = tuple(marker for _predicate, marker in predicate_markers)
    solver_symbols = tuple(sorted(set(user_symbols) | set(marker_names)))
    degree_bounds = _safe_predicate_upper_bounds(constraints)
    arithmetic = ArithmeticContext(
        backend=ArithmeticBackend.FMPQ_MPOLY,
        symbolic_variables=solver_symbols,
        output_symbols=user_symbols,
    )
    weights = compile_weight_mapping(dict(problem.weights), arithmetic)
    for predicate, marker in predicate_markers_tuple:
        positive, negative = weights.get(
            predicate,
            (arithmetic.one(), arithmetic.one()),
        )
        weights[predicate] = (
            arithmetic.multiply(positive, arithmetic.symbol(marker)),
            negative,
        )
    limits = dict(problem.internal_weight_degree_limits)
    for predicate, marker in predicate_markers_tuple:
        bound = degree_bounds.get(predicate)
        if bound is None:
            continue
        limits[marker] = (
            bound if marker not in limits else DomainExpr.minimum(limits[marker], bound)
        )
    return replace(
        problem,
        weights=dict(sorted(weights.items(), key=lambda item: str(item[0]))),
        cardinality_constraints=(),
        internal_weight_symbols=tuple(
            sorted(set(problem.internal_weight_symbols) | set(marker_names))
        ),
        internal_weight_degree_limits=tuple(sorted(limits.items())),
        decoder_spec=problem.decoder_spec.append(
            CardinalityDecoderSpec(constraints, predicate_markers_tuple)
        ),
    )


def _safe_predicate_upper_bounds(
    constraints: tuple[ReducedCardinalityConstraint, ...],
) -> dict[object, DomainExpr]:
    bounds: dict[object, list[DomainExpr]] = {}
    for constraint in constraints:
        if constraint.comparator in (Comparator.EQ, Comparator.LE):
            total = constraint.rhs
        elif constraint.comparator is Comparator.LT:
            total = constraint.rhs - 1
        else:
            continue
        coefficients: dict[object, int] = {}
        for term in constraint.terms:
            coefficients[term.predicate] = (
                coefficients.get(term.predicate, 0) + term.coefficient
            )
        if any(coefficient < 0 for coefficient in coefficients.values()):
            continue
        for predicate, coefficient in coefficients.items():
            if coefficient > 0:
                bounds.setdefault(predicate, []).append(total // coefficient)
    return {
        predicate: DomainExpr.minimum(*predicate_bounds)
        for predicate, predicate_bounds in bounds.items()
    }


def _reduced_constraint(
    constraint: LinearCardinalityConstraint,
) -> ReducedCardinalityConstraint:
    return ReducedCardinalityConstraint(
        terms=constraint.terms,
        comparator=constraint.comparator,
        rhs=DomainExpr.constant(constraint.rhs),
        modulus=constraint.modulus,
    )


def _expr(value: int | DomainExpr) -> DomainExpr:
    return value if isinstance(value, DomainExpr) else DomainExpr.constant(value)


def _exact_int(value: Fraction, label: str) -> int:
    if value.denominator != 1:
        raise ValueError(f"{label} must be integral, got {value}")
    return value.numerator


def _profile_sort_key(profile: frozenset[Literal]) -> tuple[tuple[str, bool], ...]:
    return tuple(
        sorted((str(literal.predicate), literal.positive) for literal in profile)
    )


__all__ = [
    "DecoderSpec",
    "DomainExpr",
    "ReducedProfileConstraint",
    "ReducedProblem",
    "ReducedCardinalityConstraint",
    "reduce_problem",
]
