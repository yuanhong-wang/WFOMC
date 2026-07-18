"""WFOMCS grammar transformer."""

from __future__ import annotations

from fractions import Fraction

from wfomc.cardinality_constraints import (
    CardinalityConstraints,
    CardinalityTerm,
    Comparator,
    LinearCardinalityConstraint,
)
from wfomc.evidence import Evidence
from wfomc.problem import Domain, Problem, ProblemInstance

from .cardinality import CCTransformer
from .fol import FormulaTransformer


class ProblemTransformer(FormulaTransformer, CCTransformer):
    """Transform a WFOMCS parse tree into the native ``Problem`` IR."""

    def domain_elements(self, args):
        return list(args)

    def int_domain(self, args):
        return int(args[0])

    def element(self, args):
        return args[0].value

    def set_domain(self, args):
        return {self.context.constant(i) for i in args[0]}

    def domain_name(self, args):
        return args[0].value

    def domain(self, args):
        domain_name, domain_spec = args
        if isinstance(domain_spec, int):
            domain_spec = {
                self.context.constant(f"{domain_name}{index}")
                for index in range(domain_spec)
            }
        return (domain_name, domain_spec)

    def weightings(self, args):
        return dict(args)

    def weighting(self, args):
        return (
            args[2],
            (
                Fraction(
                    Fraction(args[0]).numerator,
                    Fraction(args[0]).denominator,
                ),
                Fraction(
                    Fraction(args[1]).numerator,
                    Fraction(args[1]).denominator,
                ),
            ),
        )

    def weight(self, args):
        return str(args[0])

    def _predicate_by_name(self, name: str) -> object:
        pred = self.name2pred.get(name)
        if pred is None:
            raise ValueError(f"Predicate {name} not found")
        return pred

    def _cardinality_constraints(
        self,
        constraints: list[tuple[dict[str, object], str, object]],
    ) -> CardinalityConstraints:
        return CardinalityConstraints(
            tuple(
                self._cardinality_constraint(constraint)
                for constraint in constraints
            )
        )

    def _cardinality_constraint(
        self,
        constraint: tuple[dict[str, object], str, object],
    ) -> LinearCardinalityConstraint:
        expr, comparator, param = constraint
        return LinearCardinalityConstraint(
            terms=tuple(
                CardinalityTerm(
                    self._predicate_by_name(pred_name),
                    _exact_int(coefficient, "coefficient"),
                )
                for pred_name, coefficient in sorted(
                    expr.items(),
                    key=lambda item: str(item[0]),
                )
                if _exact_int(coefficient, "coefficient") != 0
            ),
            comparator=Comparator(str(comparator)),
            rhs=_exact_int(param, "rhs"),
        )

    def wfomcs(self, args) -> ProblemInstance:
        sentence = args[0]
        _domain_name, domain = args[1]
        weightings = args[2]
        cardinality_constraints = args[3]
        unary_evidence = args[4]

        pred_weights = {
            self._predicate_by_name(pred_name): weights
            for pred_name, weights in weightings.items()
        }
        return ProblemInstance(
            problem=Problem(
                sentence=sentence,
                weights=pred_weights,
                cardinality_constraints=self._cardinality_constraints(
                    cardinality_constraints
                ),
                evidence=Evidence(unary=unary_evidence),
            ),
            domain=Domain(frozenset(domain)),
        )


def _exact_int(value: object, label: str) -> int:
    rational = Fraction(value).limit_denominator()
    if rational.denominator != 1:
        raise ValueError(
            f"Cardinality {label} must be an integer in the current IR: {value!r}"
        )
    return rational.numerator


__all__ = [
    "ProblemTransformer",
]
