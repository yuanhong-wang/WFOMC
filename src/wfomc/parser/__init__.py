from __future__ import annotations

from .parse import (
    parse_formula,
    parse_mln_problem,
    parse_mln_problem_file,
    parse_problem,
    parse_problem_file,
)

parse_input = parse_problem_file


__all__ = [
    "parse_input",
    "parse_formula",
    "parse_mln_problem",
    "parse_mln_problem_file",
    "parse_problem",
    "parse_problem_file",
]
