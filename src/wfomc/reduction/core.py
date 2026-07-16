"""Small reduction contracts shared by the engine and algorithms."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

if TYPE_CHECKING:
    from wfomc.algo.core import AlgoOptions
    from wfomc.problem import Problem, ReducedProblem


Decoder = Callable[..., object]


ReductionResult: TypeAlias = "ReducedProblem | ProblemWithDecoder | ReducedProblems"


class ReductionFn(Protocol):
    """One logical stage transformation used by ``apply_reductions``."""

    def __call__(
        self,
        problem: "ReducedProblem",
        *,
        options: "AlgoOptions",
    ) -> ReductionResult: ...


def identity_decoder(result: object, **_: object) -> object:
    return result


@dataclass(frozen=True)
class ProblemWithDecoder:
    """One source/reduced branch plus the decoder for its raw result."""

    problem: "Problem | ReducedProblem"
    decoder: Decoder = identity_decoder


@dataclass(frozen=True)
class ReducedProblems:
    """A reduction result: one or more problems to solve and decode."""

    problems: tuple[ProblemWithDecoder, ...]

    @classmethod
    def single(
        cls,
        problem: "Problem | ReducedProblem",
        *,
        decoder: Decoder = identity_decoder,
    ) -> "ReducedProblems":
        return cls((ProblemWithDecoder(problem, decoder),))

    @classmethod
    def from_items(
        cls,
        items: Iterable[ProblemWithDecoder],
    ) -> "ReducedProblems":
        return cls(tuple(items))

    def expect_single(self) -> ProblemWithDecoder:
        if len(self.problems) != 1:
            raise ValueError(
                f"Expected exactly one reduced problem, got {len(self.problems)}"
            )
        return self.problems[0]


def compose_decoders(outer: Decoder, inner: Decoder) -> Decoder:
    def decode(result: object, **kwargs: object) -> object:
        return outer(inner(result, **kwargs), **kwargs)

    return decode


def divide_decoder(coefficient: object) -> Decoder:
    if coefficient == 1:
        return identity_decoder

    def decode(result: object, **kwargs: object) -> object:
        arithmetic = kwargs.get("arithmetic")
        if arithmetic is None:
            raise TypeError("divide decoder requires ArithmeticContext")
        return result / arithmetic.coerce(coefficient)

    return decode


def begin_reduction(problem: "Problem") -> "ReducedProblem":
    """Normalize one public source problem into logical reduction state."""

    from wfomc.fol.normal_form import normalize, validate_normal_form
    from wfomc.problem import ReducedProblem

    normal_form = normalize(
        problem.sentence,
        reserved_predicate_names=problem.declared_predicate_names(),
    )
    validate_normal_form(normal_form)
    if normal_form.requires_nonempty_domain and not problem.domain:
        raise ValueError(
            "C2 Scott abstraction requires a non-empty domain"
        )
    return ReducedProblem(
        normal_form=normal_form,
        domain=problem.domain,
        weights=problem.weights,
        cardinality_constraints=problem.cardinality_constraints,
        evidence=problem.evidence,
        circular_order_size=problem.circular_order_size,
    )


def apply_reductions(
    problem: "Problem",
    reductions: tuple[ReductionFn, ...],
    options: "AlgoOptions",
) -> ReducedProblems:
    """Apply an explicit logical reduction sequence and compose its decoders."""

    branches = (ProblemWithDecoder(begin_reduction(problem), identity_decoder),)
    for reduce_fn in reductions:
        next_branches: list[ProblemWithDecoder] = []
        for branch in branches:
            result = reduce_fn(branch.problem, options=options)
            if isinstance(result, ReducedProblems):
                reduced = result
            elif isinstance(result, ProblemWithDecoder):
                reduced = ReducedProblems((result,))
            else:
                reduced = ReducedProblems.single(result)
            for item in reduced.problems:
                next_branches.append(
                    ProblemWithDecoder(
                        item.problem,
                        compose_decoders(branch.decoder, item.decoder),
                    )
                )
        branches = tuple(next_branches)
    return ReducedProblems(branches)


__all__ = [
    "Decoder",
    "apply_reductions",
    "begin_reduction",
    "ProblemWithDecoder",
    "ReductionFn",
    "ReductionResult",
    "ReducedProblems",
    "compose_decoders",
    "divide_decoder",
    "identity_decoder",
]
