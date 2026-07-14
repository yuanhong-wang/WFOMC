"""Pure problem reductions."""

from __future__ import annotations

from .core import (
    ProblemWithDecoder,
    ReducedProblems,
    begin_reduction,
    compose_decoders,
    divide_decoder,
    identity_decoder,
    apply_reductions,
)
from .reduce_cardinality_constraints import reduce_cardinality_constraints
from .reduce_counting_quantifiers import (
    reduce_counting_quantifiers,
)
from .reduce_existential_quantifiers import reduce_existential_quantifiers
from .reduce_unary_evidence import (
    UnaryCcsEncoding,
    build_profile_capacity_constraint,
    build_unary_ccs_encoding,
    reduce_unary_evidence_to_cardinality_constraints,
    reduce_unary_evidence_to_profile_capacity,
)

__all__ = [
    "ProblemWithDecoder",
    "ReducedProblems",
    "begin_reduction",
    "compose_decoders",
    "divide_decoder",
    "identity_decoder",
    "apply_reductions",
    "reduce_cardinality_constraints",
    "reduce_counting_quantifiers",
    "reduce_existential_quantifiers",
    "UnaryCcsEncoding",
    "build_profile_capacity_constraint",
    "build_unary_ccs_encoding",
    "reduce_unary_evidence_to_cardinality_constraints",
    "reduce_unary_evidence_to_profile_capacity",
]
