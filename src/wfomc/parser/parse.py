"""Parser entry points."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lark import Lark, UnexpectedInput

from wfomc.parser.grammar.mln import MLN_GRAMMAR
from wfomc.parser.grammar.wfomcs import WFOMCS_GRAMMAR
from wfomc.parser.transformers.fol import parse as parse_typed_formula
from wfomc.parser.transformers.mln import MLNTransformer
from wfomc.parser.transformers.wfomcs import ProblemTransformer
from wfomc.problem import Domain, Problem, ProblemInstance


def parse_formula(text: str) -> object:
    return parse_typed_formula(text)


def parse_problem(text: str) -> ProblemInstance:
    parser = Lark(WFOMCS_GRAMMAR, start="wfomcs")
    try:
        tree = parser.parse(text)
    except UnexpectedInput:
        return ProblemInstance(Problem(sentence=parse_formula(text)), Domain())
    return ProblemTransformer().transform(tree)


def parse_problem_file(path: str | Path) -> ProblemInstance:
    source_path = Path(path)
    if source_path.suffix == ".mln":
        return parse_mln_problem_file(source_path)
    parsed = parse_problem(source_path.read_text())
    return _with_source_path(parsed, source_path)


def parse_mln_problem(text: str) -> ProblemInstance:
    parser = Lark(MLN_GRAMMAR, start="mln")
    tree = parser.parse(text)
    return MLNTransformer().transform(tree)


def parse_mln_problem_file(path: str | Path) -> ProblemInstance:
    source_path = Path(path)
    parsed = parse_mln_problem(source_path.read_text())
    return _with_source_path(parsed, source_path)


def _with_source_path(
    instance: ProblemInstance,
    path: Path,
) -> ProblemInstance:
    return replace(instance, source_path=str(path))


__all__ = [
    "parse_formula",
    "parse_mln_problem",
    "parse_mln_problem_file",
    "parse_problem",
    "parse_problem_file",
]
