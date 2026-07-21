"""Native public package surface for WFOMC."""

from __future__ import annotations

import logging

from wfomc.algo.core import AlgoMaturity, AlgoName, AlgoOptions
from wfomc.engine import (
    CompiledProblem,
    ProblemExecution,
    RuntimeContext,
    RuntimeOptions,
    compile_problem,
    instantiate_problem,
    solve,
)
from wfomc.cardinality_constraints import (
    CardinalityConstraints,
    CardinalityTerm,
    Comparator,
    LinearCardinalityConstraint,
)
from wfomc.evidence import (
    Evidence,
    GroundUnaryLiteral,
    UnaryEvidence,
)
from wfomc.fol.grounding import LinearOrderEncoding
from wfomc.options import EvidenceStrategy, ExistentialStrategy, WeightOptions
from wfomc.parser import (
    parse_formula,
    parse_mln_problem,
    parse_mln_problem_file,
    parse_problem,
    parse_problem_file,
)
from wfomc.problem import Domain, Problem, ProblemInstance
from wfomc.result import WFOMCResult

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AlgoName",
    "AlgoMaturity",
    "AlgoOptions",
    "CardinalityConstraints",
    "CardinalityTerm",
    "Comparator",
    "CompiledProblem",
    "Domain",
    "Evidence",
    "EvidenceStrategy",
    "ExistentialStrategy",
    "GroundUnaryLiteral",
    "LinearCardinalityConstraint",
    "LinearOrderEncoding",
    "Problem",
    "ProblemExecution",
    "ProblemInstance",
    "RuntimeContext",
    "RuntimeOptions",
    "UnaryEvidence",
    "WFOMCResult",
    "WeightOptions",
    "compile_problem",
    "instantiate_problem",
    "parse_formula",
    "parse_mln_problem",
    "parse_mln_problem_file",
    "parse_problem",
    "parse_problem_file",
    "solve",
]
