"""Native public package surface for WFOMC."""

from __future__ import annotations

import logging

from wfomc.api import (
    AlgoName,
    AlgoMaturity,
    AlgoOptions,
    CompileArtifacts,
    EvidenceStrategy,
    ExistentialStrategy,
    RuntimeContext,
    RuntimeOptions,
    WFOMCResult,
    compile_problem,
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
from wfomc.parser import (
    parse_formula,
    parse_input,
    parse_mln_problem,
    parse_mln_problem_file,
    parse_problem,
    parse_problem_file,
)
from wfomc.problem import Problem

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AlgoName",
    "AlgoMaturity",
    "AlgoOptions",
    "CardinalityConstraints",
    "CardinalityTerm",
    "Comparator",
    "CompileArtifacts",
    "Evidence",
    "EvidenceStrategy",
    "ExistentialStrategy",
    "GroundUnaryLiteral",
    "LinearCardinalityConstraint",
    "Problem",
    "RuntimeContext",
    "RuntimeOptions",
    "UnaryEvidence",
    "WFOMCResult",
    "compile_problem",
    "parse_formula",
    "parse_input",
    "parse_mln_problem",
    "parse_mln_problem_file",
    "parse_problem",
    "parse_problem_file",
    "solve",
]
