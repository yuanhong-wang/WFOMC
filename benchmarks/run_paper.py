#!/usr/bin/env python3
"""Run the fixed Core domain grids used by the paper experiments."""

from __future__ import annotations

from functools import lru_cache, partial
from math import factorial
from pathlib import Path
from typing import Sequence

from benchmarks.cases import (
    BenchmarkCase,
    _direct_c2_properly_coloured_undirected_regular_problem,
    _properly_coloured_undirected_regular_reduction_problem,
    core_case,
)
from benchmarks.run import parse_args, run_catalog, worker_main
from wfomc import (
    Domain,
    Problem,
    ProblemInstance,
    parse_formula,
    parse_problem_file,
)
from wfomc.fol import Formula, context_for


ROOT = Path(__file__).resolve().parents[1]
_TYPED_SPARSE_TYPE_COUNT = 8
_TYPED_SPARSE_ASYMMETRIC_EDGES = (
    (0, 1),
    (0, 4),
    (0, 5),
    (0, 6),
    (1, 3),
    (1, 4),
    (1, 6),
    (2, 3),
    (2, 4),
    (2, 6),
    (3, 5),
    (3, 6),
    (4, 6),
    (4, 7),
    (5, 6),
    (5, 7),
    (6, 7),
)
_TYPED_SPARSE_SHAPES = {
    "typed-path-relation-k8": "path",
    "typed-tree-relation-k8": "tree",
    "typed-cycle-relation-k8": "cycle",
    "typed-asymmetric-relation-k8": "asymmetric",
}
_COLOURED_REGULAR_CONFIGURATIONS = {
    "properly-3-coloured-undirected-3-regular": (3, 3),
    "properly-4-coloured-undirected-3-regular": (4, 3),
    "properly-5-coloured-undirected-3-regular": (5, 3),
    "properly-4-coloured-undirected-2-regular": (4, 2),
    "properly-4-coloured-undirected-4-regular": (4, 4),
}
_FRIENDS_SMOKERS = "friends-smokers"
_RELATIONAL_MLN_MODELS = {
    "academic-advising": "models/paper/academic-advising.mln",
    "id2-gene-regulation": "models/paper/id2-gene-regulation.mln",
    "imdb-worked-under-fo2": "models/paper/imdb-worked-under-fo2.mln",
    "webkb-link-classification": (
        "models/paper/webkb-link-classification.mln"
    ),
}
PAPER_DOMAIN_GRIDS: tuple[tuple[str, tuple[int, ...]], ...] = (
    (
        "permutations",
        (20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 125),
    ),
    (
        "undirected-2-regular",
        (20, 40, 60, 80, 100, 120, 140, 160, 180, 200),
    ),
    (
        "undirected-3-regular",
        (20, 30, 40, 50, 60, 70, 80, 90, 100),
    ),
    (
        "undirected-4-regular",
        (12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50, 52, 56, 60),
    ),
    (
        "properly-2-coloured-graph",
        (100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000),
    ),
    (
        "properly-3-coloured-graph",
        (50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300),
    ),
    (
        "properly-4-coloured-graph",
        (20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150),
    ),
    (
        "properly-5-coloured-graph",
        (16, 24, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75),
    ),
    (
        "derangements",
        (50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300),
    ),
    (
        "endofunctions",
        (40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240),
    ),
    (
        "loopless-digraph-without-isolates",
        (50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600),
    ),
    (
        "2-edge-disjoint-perfect-matchings",
        (20, 40, 60, 80, 100, 150, 200, 250, 300, 400),
    ),
    (
        "3-edge-disjoint-perfect-matchings",
        (10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
    ),
    (
        "4-edge-disjoint-perfect-matchings",
        (6, 8, 10, 12, 14, 16),
    ),
    (
        "typed-path-relation-k8",
        (20, 40, 60, 80, 100, 120, 160, 200, 250, 300, 400, 500),
    ),
    (
        "typed-tree-relation-k8",
        (20, 40, 60, 80, 100, 120, 160, 200, 250, 300, 400, 500),
    ),
    (
        "typed-cycle-relation-k8",
        (20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200),
    ),
    (
        "typed-asymmetric-relation-k8",
        (20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 300),
    ),
    (
        "properly-3-coloured-undirected-3-regular",
        (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
    ),
    (
        "properly-4-coloured-undirected-3-regular",
        (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
    ),
    (
        "properly-5-coloured-undirected-3-regular",
        (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
    ),
    (
        "properly-4-coloured-undirected-2-regular",
        (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
    ),
    (
        "properly-4-coloured-undirected-4-regular",
        (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
    ),
    (
        _FRIENDS_SMOKERS,
        (20, 40, 60, 80, 100, 120, 140, 160, 180, 200),
    ),
    (
        "academic-advising",
        (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
    ),
    (
        "id2-gene-regulation",
        (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22),
    ),
    (
        "imdb-worked-under-fo2",
        (
            2, 4, 6, 8, 12, 16, 24, 32, 40, 48,
            56, 64, 72, 80, 96, 112, 128, 144, 160,
        ),
    ),
    (
        "webkb-link-classification",
        (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
    ),
)


def _typed_sparse_edges(shape: str) -> tuple[tuple[int, int], ...]:
    type_count = _TYPED_SPARSE_TYPE_COUNT
    if shape == "path":
        return tuple((index, index + 1) for index in range(type_count - 1))
    if shape == "tree":
        return tuple(((index - 1) // 2, index) for index in range(1, type_count))
    if shape == "cycle":
        return (
            *((index, index + 1) for index in range(type_count - 1)),
            (type_count - 1, 0),
        )
    if shape == "asymmetric":
        return _TYPED_SPARSE_ASYMMETRIC_EDGES
    raise ValueError(f"unknown typed sparse-interaction shape: {shape}")


@lru_cache(maxsize=None)
def _typed_sparse_sentence(shape: str) -> Formula:
    """Build one fixed eight-type interaction sentence for every domain."""

    types = tuple(
        f"C{index + 1}" for index in range(_TYPED_SPARSE_TYPE_COUNT)
    )
    clauses = [
        rf"(\forall X: ({' | '.join(f'{name}(X)' for name in types)}))",
        *(
            rf"(\forall X: (~{left}(X) | ~{right}(X)))"
            for index, left in enumerate(types)
            for right in types[index + 1 :]
        ),
        r"(\forall X: (~E(X,X)))",
        r"(\forall X: (\forall Y: (E(X,Y) -> E(Y,X))))",
    ]
    allowed_pairs = []
    for left, right in _typed_sparse_edges(shape):
        allowed_pairs.extend(
            (
                f"({types[left]}(X) & {types[right]}(Y))",
                f"({types[right]}(X) & {types[left]}(Y))",
            )
        )
    clauses.append(
        r"(\forall X: (\forall Y: (E(X,Y) -> ("
        + " | ".join(allowed_pairs)
        + "))))"
    )
    return parse_formula(" & ".join(clauses))


def _instantiate_paper_problem(
    problem: Problem,
    domain_size: int,
    *,
    source_path: Path | None = None,
) -> ProblemInstance:
    formula_context = context_for(problem.sentence)
    domain = Domain(
        frozenset(
            formula_context.constant(f"d{index}")
            for index in range(domain_size)
        )
    )
    return ProblemInstance(
        problem,
        domain,
        source_path=str(source_path) if source_path is not None else None,
    )


def _typed_sparse_problem(shape: str, domain_size: int) -> ProblemInstance:
    return _instantiate_paper_problem(
        Problem(sentence=_typed_sparse_sentence(shape)),
        domain_size,
    )


@lru_cache(maxsize=None)
def _model_problem(relative_path: str) -> Problem:
    """Parse the domain-independent part of a paper model only once."""

    return parse_problem_file(ROOT / relative_path).problem


def _model_problem_instance(
    relative_path: str,
    domain_size: int,
) -> ProblemInstance:
    return _instantiate_paper_problem(
        _model_problem(relative_path),
        domain_size,
        source_path=ROOT / relative_path,
    )


def _friends_smokers_problem(domain_size: int) -> ProblemInstance:
    return _model_problem_instance(
        "models/friends-smokes.wfomcs",
        domain_size,
    )


def _paper_case(family: str, domain_size: int) -> BenchmarkCase:
    shape = _TYPED_SPARSE_SHAPES.get(family)
    if shape is not None:
        return BenchmarkCase(
            key=f"core/{family}/n{domain_size}",
            family=family,
            category="core",
            domain_size=domain_size,
            purposes=frozenset(("paper", "typed-sparse-interaction")),
            _builder=partial(_typed_sparse_problem, shape),
        )
    coloured_regular = _COLOURED_REGULAR_CONFIGURATIONS.get(family)
    if coloured_regular is not None:
        colour_count, degree = coloured_regular
        return BenchmarkCase(
            key=(
                f"core/{family}/fo2-cardinality-reduction/n{domain_size}"
            ),
            family=family,
            category="core",
            domain_size=domain_size,
            variant="fo2-cardinality-reduction",
            purposes=frozenset(
                ("paper", "combined-structure", "coloured-regular")
            ),
            correction_divisor=factorial(degree) ** domain_size,
            _builder=partial(
                _properly_coloured_undirected_regular_reduction_problem,
                colour_count=colour_count,
                degree=degree,
            ),
            _original_c2_builder=partial(
                _direct_c2_properly_coloured_undirected_regular_problem,
                colour_count=colour_count,
                degree=degree,
            ),
        )
    if family == _FRIENDS_SMOKERS:
        return BenchmarkCase(
            key=f"core/{family}/n{domain_size}",
            family=family,
            category="core",
            domain_size=domain_size,
            purposes=frozenset(("paper", "relational-model")),
            _builder=_friends_smokers_problem,
        )
    relational_mln = _RELATIONAL_MLN_MODELS.get(family)
    if relational_mln is not None:
        return BenchmarkCase(
            key=f"core/{family}/n{domain_size}",
            family=family,
            category="core",
            domain_size=domain_size,
            purposes=frozenset(
                ("paper", "relational-model", "bp-scaling")
            ),
            _builder=partial(_model_problem_instance, relational_mln),
        )
    return core_case(family, domain_size)


def build_paper_benchmark_case(
    family: str,
    domain_size: int,
) -> BenchmarkCase:
    """Build one paper-family case at an explicitly requested domain size."""

    return _paper_case(family, domain_size)


PAPER_BENCHMARK_CASES: tuple[BenchmarkCase, ...] = tuple(
    _paper_case(family, domain_size)
    for family, domain_sizes in PAPER_DOMAIN_GRIDS
    for domain_size in domain_sizes
)
_PAPER_CASES_BY_KEY = {case.key: case for case in PAPER_BENCHMARK_CASES}


def paper_benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Return the deterministic fixed-grid paper benchmark catalog."""

    return PAPER_BENCHMARK_CASES


def paper_benchmark_case(key: str) -> BenchmarkCase:
    """Look up one concrete paper case by its stable key."""

    try:
        return _PAPER_CASES_BY_KEY[key]
    except KeyError as error:
        raise KeyError(f"unknown Core paper case: {key}") from error


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(
        argv,
        description=__doc__,
        default_out=ROOT / "benchmark-results" / "paper-core",
        default_protocol="cold-stop",
        default_timeout=300.0,
        default_repetitions=3,
    )
    if args.worker:
        if args.algorithm is None or not args.case:
            raise SystemExit("worker mode requires --algorithm and --case")
        return worker_main(args, case_lookup=paper_benchmark_case)
    if args.protocol != "cold-stop":
        raise SystemExit("the paper benchmark requires --protocol cold-stop")
    return run_catalog(
        paper_benchmark_cases(),
        args,
        worker_module="benchmarks.run_paper",
    )


if __name__ == "__main__":
    raise SystemExit(main())
