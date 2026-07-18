"""Canonical benchmark-case catalog.

This module migrates only the problem families and domain-size grids from the
historical benchmark scripts.  It deliberately contains no runner, baseline
implementation, timing policy, or result-file handling.  Every case builds the
current typed :class:`wfomc.ProblemInstance`, so benchmark runners can remain
small and choose their own algorithms and measurement protocol.

Public interface:

``BenchmarkCase``
    Immutable case metadata plus ``build_problem()``.
``benchmark_suite_names()``
    Stable names accepted by ``benchmark_cases()``.
``benchmark_cases(suite)``
    Deterministic tuple of cases in a suite.
``benchmark_case(key)``
    Look up one concrete case by its stable key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import partial
from math import comb
from typing import Callable, Mapping

from wfomc import (
    AlgoName,
    CardinalityConstraints,
    CardinalityTerm,
    Comparator,
    Domain,
    LinearCardinalityConstraint,
    Problem,
    ProblemInstance,
    parse_formula,
)
from wfomc.fol import Formula, context_for, predicates


Weight = tuple[int, int]
FormulaDefinition = tuple[str, Mapping[str, Weight]]
FormulaFactory = Callable[[], FormulaDefinition]
ProblemBuilder = Callable[[int], ProblemInstance]
ConstraintSpec = tuple[str, Comparator, int]
UnaryBounds = tuple[tuple[Comparator, int], ...]


@dataclass(frozen=True, kw_only=True)
class BenchmarkCase:
    """One concrete benchmark input, independent of any timing runner.

    ``correction_divisor`` records the multiplicity introduced by a hand
    reduction.  A runner comparing mathematical answers should divide the raw
    WFOMC result by it; ``build_problem()`` itself never changes solver output.
    ``comparison_group`` links alternative encodings of the same problem.
    """

    key: str
    family: str
    category: str
    domain_size: int
    default_algorithm: AlgoName
    variant: str = "default"
    purposes: frozenset[str] = frozenset()
    correction_divisor: int = 1
    comparison_group: str | None = None
    _builder: ProblemBuilder = field(repr=False, compare=False)

    def build_problem(self) -> ProblemInstance:
        """Build a fresh typed problem/domain instance for this case."""

        return self._builder(self.domain_size)


def _pairwise_negative(symbols: list[str], variable: str = "x") -> list[str]:
    return [
        f"(~{left}({variable}) | ~{right}({variable}))"
        for index, left in enumerate(symbols)
        for right in symbols[index + 1 :]
    ]


def _pairwise_negative_binary(symbols: list[str]) -> list[str]:
    return [
        f"(~{left}(x,y) | ~{right}(x,y))"
        for index, left in enumerate(symbols)
        for right in symbols[index + 1 :]
    ]


def _k_regular_definition(k: int) -> FormulaDefinition:
    """Hand reduction of undirected k-regular graphs used by the FO2 suite."""

    if k < 1:
        raise ValueError("k must be positive")

    skolem = [f"S{i}" for i in range(1, k + 1)]
    canonical = [f"C{j}" for j in range(k + 1)]
    partitions = [f"F{i}" for i in range(1, k + 1)]

    canonical_clauses: list[str] = []
    for j, name in enumerate(canonical):
        literals = [f"{slot}(x)" for slot in skolem[:j]]
        literals.extend(f"~{slot}(x)" for slot in skolem[j:])
        canonical_clauses.append(
            f"({name}(x) <-> ({' & '.join(literals)}))"
        )

    clauses = [
        "~E(x,x)",
        "(E(x,y) -> E(y,x))",
        "(E(x,y) <-> F(x,y))",
        *(
            f"({partition}(x,y) -> {slot}(x))"
            for partition, slot in zip(partitions, skolem)
        ),
        f"(F(x,y) <-> ({' | '.join(f'{name}(x,y)' for name in partitions)}))",
        *_pairwise_negative_binary(partitions),
        *canonical_clauses,
        f"({' | '.join(f'{name}(x)' for name in canonical)})",
    ]
    weights = {name: (1, -1) for name in skolem}
    weights.update({name: (comb(k, j), 1) for j, name in enumerate(canonical)})
    weights.update({name: (1, 1) for name in ("E", "F", *partitions)})
    return " & ".join(f"({clause})" for clause in clauses), weights


def _k_coloured_definition(k: int) -> FormulaDefinition:
    if k < 2:
        raise ValueError("k must be at least 2")

    colours = [f"C{i}" for i in range(1, k + 1)]
    different_colour = " | ".join(
        f"({colour}(x) & ~{colour}(y))" for colour in colours
    )
    clauses = [
        "~E(x,x)",
        "(E(x,y) -> E(y,x))",
        f"({' | '.join(f'{name}(x)' for name in colours)})",
        *_pairwise_negative(colours),
        f"(E(x,y) -> ({different_colour}))",
    ]
    weights = {name: (1, 1) for name in (*colours, "E")}
    return " & ".join(f"({clause})" for clause in clauses), weights


def _derangements_definition() -> FormulaDefinition:
    return (
        "~F(x,x) & (S1(x) | ~F(x,y)) & (S2(x) | ~F(y,x))",
        {"S1": (1, -1), "S2": (1, -1), "F": (1, 1)},
    )


def _k_matchings_definition(k: int) -> FormulaDefinition:
    if k < 1:
        raise ValueError("k must be positive")

    skolem = [f"S{i}" for i in range(1, k + 1)]
    edges = [f"E{i}" for i in range(1, k + 1)]
    clauses = [
        *(f"~{name}(x,x)" for name in edges),
        *(f"({name}(x,y) -> {name}(y,x))" for name in edges),
        *(
            f"({slot}(x) | ~{edge}(x,y))"
            for slot, edge in zip(skolem, edges)
        ),
        *(
            f"({left}(x,y) -> ~{right}(y,x))"
            for index, left in enumerate(edges)
            for right in edges[index + 1 :]
        ),
    ]
    weights = {name: (1, -1) for name in skolem}
    weights.update({name: (1, 1) for name in edges})
    return " & ".join(f"({clause})" for clause in clauses), weights


def _row_column_definition() -> FormulaDefinition:
    return (
        "(Sx(x) | ~P(x,y)) & (Sy(y) | ~P(x,y))",
        {"Sx": (1, -1), "Sy": (1, -1), "P": (1, 1)},
    )


def _total_mappings_definition() -> FormulaDefinition:
    return "S(x) | ~F(x,y)", {"S": (1, -1), "F": (1, 1)}


def _no_isolated_digraph_definition() -> FormulaDefinition:
    return (
        "~E(x,x) & (S(x) | E(x,y) | E(y,x))",
        {"S": (1, -1), "E": (1, 1)},
    )


def _three_regular_four_coloured_definition() -> FormulaDefinition:
    colours = [f"Col{i}" for i in range(1, 5)]
    skolem = [f"S{i}" for i in range(1, 4)]
    canonical = [f"T{j}" for j in range(4)]
    partitions = [f"F{i}" for i in range(1, 4)]

    colour_clauses = [f"({' | '.join(f'{name}(x)' for name in colours)})"]
    colour_clauses.extend(_pairwise_negative(colours))
    colour_clauses.extend(
        f"(~E(x,y) | ~{name}(x) | ~{name}(y))" for name in colours
    )

    canonical_clauses: list[str] = []
    for j, name in enumerate(canonical):
        literals = [f"{slot}(x)" for slot in skolem[:j]]
        literals.extend(f"~{slot}(x)" for slot in skolem[j:])
        canonical_clauses.append(
            f"({name}(x) <-> ({' & '.join(literals)}))"
        )

    clauses = [
        "~E(x,x)",
        "(E(x,y) -> E(y,x))",
        *colour_clauses,
        "(E(x,y) <-> F(x,y))",
        *(
            f"({partition}(x,y) -> {slot}(x))"
            for partition, slot in zip(partitions, skolem)
        ),
        f"(F(x,y) <-> ({' | '.join(f'{name}(x,y)' for name in partitions)}))",
        *_pairwise_negative_binary(partitions),
        *canonical_clauses,
        f"({' | '.join(f'{name}(x)' for name in canonical)})",
    ]
    weights = {name: (1, 1) for name in colours}
    weights.update({name: (1, -1) for name in skolem})
    weights.update({name: (comb(3, j), 1) for j, name in enumerate(canonical)})
    weights.update({name: (1, 1) for name in ("E", "F", *partitions)})
    return " & ".join(f"({clause})" for clause in clauses), weights


def _directed_three_regular_definition() -> FormulaDefinition:
    out_skolem = [f"OutS{i}" for i in range(1, 4)]
    out_canonical = [f"OutT{j}" for j in range(4)]
    in_skolem = [f"InS{i}" for i in range(1, 4)]
    in_canonical = [f"InT{j}" for j in range(4)]
    out_partitions = [f"OutF{i}" for i in range(1, 4)]
    in_partitions = [f"InF{i}" for i in range(1, 4)]

    def canonical_clauses(
        names: list[str], slots: list[str], variable: str
    ) -> list[str]:
        result: list[str] = []
        for j, name in enumerate(names):
            literals = [f"{slot}({variable})" for slot in slots[:j]]
            literals.extend(f"~{slot}({variable})" for slot in slots[j:])
            result.append(f"({name}({variable}) <-> ({' & '.join(literals)}))")
        return result

    clauses = [
        "~E(x,x)",
        f"(E(x,y) <-> ({' | '.join(f'{name}(x,y)' for name in out_partitions)}))",
        *(
            f"({partition}(x,y) -> {slot}(x))"
            for partition, slot in zip(out_partitions, out_skolem)
        ),
        *_pairwise_negative_binary(out_partitions),
        *canonical_clauses(out_canonical, out_skolem, "x"),
        f"({' | '.join(f'{name}(x)' for name in out_canonical)})",
        f"(E(x,y) <-> ({' | '.join(f'{name}(x,y)' for name in in_partitions)}))",
        *(
            f"({partition}(x,y) -> {slot}(y))"
            for partition, slot in zip(in_partitions, in_skolem)
        ),
        *_pairwise_negative_binary(in_partitions),
        *canonical_clauses(in_canonical, in_skolem, "x"),
        f"({' | '.join(f'{name}(x)' for name in in_canonical)})",
    ]
    weights = {name: (1, -1) for name in (*out_skolem, *in_skolem)}
    weights.update(
        {name: (comb(3, j), 1) for j, name in enumerate(out_canonical)}
    )
    weights.update(
        {name: (comb(3, j), 1) for j, name in enumerate(in_canonical)}
    )
    weights.update(
        {
            name: (1, 1)
            for name in ("E", *out_partitions, *in_partitions)
        }
    )
    return " & ".join(f"({clause})" for clause in clauses), weights


def _typed_problem(
    sentence: object,
    domain_size: int,
    weights_by_name: Mapping[str, Weight] | None = None,
    constraint_specs: tuple[ConstraintSpec, ...] = (),
) -> ProblemInstance:
    if not isinstance(sentence, Formula):
        raise TypeError("benchmark formula parser did not return a Formula")

    declared_by_name = {predicate.name: predicate for predicate in predicates(sentence)}
    unknown_weights = set(weights_by_name or ()) - declared_by_name.keys()
    unknown_constraints = {
        name for name, _comparator, _rhs in constraint_specs
    } - declared_by_name.keys()
    if unknown_weights or unknown_constraints:
        unknown = sorted(unknown_weights | unknown_constraints)
        raise ValueError(f"benchmark metadata refers to undeclared predicates: {unknown}")

    formula_context = context_for(sentence)
    domain = frozenset(
        formula_context.constant(f"d{index}") for index in range(domain_size)
    )
    weights = {
        declared_by_name[name]: weight
        for name, weight in (weights_by_name or {}).items()
    }
    constraints = CardinalityConstraints(
        tuple(
            LinearCardinalityConstraint(
                terms=(CardinalityTerm(declared_by_name[name]),),
                comparator=comparator,
                rhs=rhs,
            )
            for name, comparator, rhs in constraint_specs
        )
    )
    return ProblemInstance(
        Problem(
            sentence=sentence,
            weights=weights,
            cardinality_constraints=constraints,
        ),
        Domain(domain),
    )


def _matrix_problem(
    domain_size: int,
    definition: FormulaFactory,
    constraint_factory: Callable[[int], tuple[ConstraintSpec, ...]] | None = None,
) -> ProblemInstance:
    matrix, weights = definition()
    matrix = _parser_variables(matrix)
    sentence = parse_formula(rf"\forall X: (\forall Y: ({matrix}))")
    constraint_specs = constraint_factory(domain_size) if constraint_factory else ()
    return _typed_problem(sentence, domain_size, weights, constraint_specs)


def _c2_three_regular_problem(domain_size: int) -> ProblemInstance:
    sentence = parse_formula(
        r"(\forall X: (~E(X,X))) & "
        r"(\forall X: (\forall Y: (E(X,Y) -> E(Y,X)))) & "
        r"(\forall X: (\exists_=3 Y: E(X,Y)))"
    )
    return _typed_problem(sentence, domain_size)


def _c2_three_coloured_regular_problem(domain_size: int) -> ProblemInstance:
    colours = ["C1", "C2", "C3"]
    clauses = [
        "~E(x,x)",
        "(E(x,y) -> E(y,x))",
        f"({' | '.join(f'{name}(x)' for name in colours)})",
        *_pairwise_negative(colours),
        *(f"(~E(x,y) | ~{name}(x) | ~{name}(y))" for name in colours),
    ]
    universal_matrix = " & ".join(f"({clause})" for clause in clauses)
    universal_matrix = _parser_variables(universal_matrix)
    universal = rf"\forall X: (\forall Y: ({universal_matrix}))"
    count = r"\forall X: (\exists_=3 Y: E(X,Y))"
    return _typed_problem(
        parse_formula(f"({universal}) & ({count})"),
        domain_size,
    )


def _c2_directed_three_in_three_out_problem(domain_size: int) -> ProblemInstance:
    sentence = parse_formula(
        r"(\forall X: (~R(X,X))) & "
        r"(\forall X: (\exists_=3 Y: R(X,Y))) & "
        r"(\forall X: (\exists_=3 Y: R(Y,X)))"
    )
    return _typed_problem(sentence, domain_size)


def _three_regular_four_coloured_problem(domain_size: int) -> ProblemInstance:
    return _matrix_problem(
        domain_size,
        _three_regular_four_coloured_definition,
        lambda n: (("F", Comparator.EQ, 3 * n),),
    )


def _directed_three_regular_problem(domain_size: int) -> ProblemInstance:
    return _matrix_problem(
        domain_size,
        _directed_three_regular_definition,
        lambda n: (("E", Comparator.EQ, 3 * n),),
    )


def _parser_variables(matrix: str) -> str:
    """Convert source-suite lowercase variables to the current parser syntax."""

    matrix = re.sub(r"\bx\b", "X", matrix)
    return re.sub(r"\by\b", "Y", matrix)


def _exact_half(domain_size: int) -> UnaryBounds:
    return ((Comparator.EQ, domain_size // 2),)


def _middle_interval(domain_size: int) -> UnaryBounds:
    return (
        (Comparator.GE, domain_size // 3),
        (Comparator.LE, (2 * domain_size) // 3),
    )


def _exact_quarter(domain_size: int) -> UnaryBounds:
    return ((Comparator.EQ, domain_size // 4),)


def _colour_interval(domain_size: int) -> UnaryBounds:
    return (
        (Comparator.GE, domain_size // 5),
        (Comparator.LE, domain_size // 3),
    )


def _unary_problem(
    domain_size: int,
    definition: FormulaFactory,
    predicate: str,
    variant: str,
    exact: Callable[[int], UnaryBounds],
    interval: Callable[[int], UnaryBounds],
) -> ProblemInstance:
    if variant == "unconstrained":
        constraint_factory = None
    elif variant == "exact":
        constraint_factory = exact
    elif variant == "interval":
        constraint_factory = interval
    else:
        raise ValueError(f"unknown unary benchmark variant: {variant}")

    def constraints(domain: int) -> tuple[ConstraintSpec, ...]:
        assert constraint_factory is not None
        return tuple(
            (predicate, comparator, rhs)
            for comparator, rhs in constraint_factory(domain)
        )

    return _matrix_problem(
        domain_size,
        definition,
        constraints if constraint_factory is not None else None,
    )


def _case(
    *,
    key: str,
    family: str,
    category: str,
    domain_size: int,
    builder: ProblemBuilder,
    algorithm: AlgoName,
    variant: str = "default",
    purposes: frozenset[str] = frozenset(),
    correction_divisor: int = 1,
    comparison_group: str | None = None,
) -> BenchmarkCase:
    return BenchmarkCase(
        key=key,
        family=family,
        category=category,
        domain_size=domain_size,
        default_algorithm=algorithm,
        variant=variant,
        purposes=purposes,
        correction_divisor=correction_divisor,
        comparison_group=comparison_group,
        _builder=builder,
    )


_ROW_COLUMN = _row_column_definition
_DERANGEMENTS = _derangements_definition
_TOTAL_MAPPINGS = _total_mappings_definition
_NO_ISOLATED_DIGRAPH = _no_isolated_digraph_definition
_TWO_REGULAR = partial(_k_regular_definition, 2)
_THREE_REGULAR = partial(_k_regular_definition, 3)
_FOUR_REGULAR = partial(_k_regular_definition, 4)
_TWO_COLOURED = partial(_k_coloured_definition, 2)
_THREE_COLOURED = partial(_k_coloured_definition, 3)
_FOUR_COLOURED = partial(_k_coloured_definition, 4)
_FIVE_COLOURED = partial(_k_coloured_definition, 5)
_TWO_MATCHINGS = partial(_k_matchings_definition, 2)
_THREE_MATCHINGS = partial(_k_matchings_definition, 3)
_FOUR_MATCHINGS = partial(_k_matchings_definition, 4)


def _core_case(family: str, domain_size: int, definition: FormulaFactory) -> BenchmarkCase:
    return _case(
        key=f"core/{family}/n{domain_size}",
        family=family,
        category="core",
        domain_size=domain_size,
        builder=partial(_matrix_problem, definition=definition),
        algorithm=AlgoName.FASTV2,
    )


_CORE_SMOKE_ENTRIES: tuple[tuple[str, int, FormulaFactory], ...] = (
    ("row-column", 8, _ROW_COLUMN),
    ("derangements", 8, _DERANGEMENTS),
    ("2-coloured", 8, _TWO_COLOURED),
    ("no-isolated-digraph", 8, _NO_ISOLATED_DIGRAPH),
)

_CORE_MAIN_ENTRIES: tuple[tuple[str, int, FormulaFactory], ...] = (
    *(("row-column", n, _ROW_COLUMN) for n in (75, 100, 150)),
    *(("3-regular", n, _THREE_REGULAR) for n in (30, 60, 100)),
    *(("4-coloured", n, _FOUR_COLOURED) for n in (100, 200, 300)),
    *(("derangements", n, _DERANGEMENTS) for n in (100, 200, 300)),
    *(("3-matchings", n, _THREE_MATCHINGS) for n in (10, 20, 30, 40)),
)

_CORE_EXTRA_ENTRIES: tuple[tuple[str, int, FormulaFactory], ...] = (
    *(("row-column", n, _ROW_COLUMN) for n in (50, 125, 200)),
    *(("2-regular", n, _TWO_REGULAR) for n in (30, 60, 100)),
    *(("4-regular", n, _FOUR_REGULAR) for n in (20, 40, 60)),
    *(("2-coloured", n, _TWO_COLOURED) for n in (100, 200, 300)),
    *(("3-coloured", n, _THREE_COLOURED) for n in (100, 200, 300)),
    *(("5-coloured", n, _FIVE_COLOURED) for n in (75, 150, 225)),
    *(("derangements", n, _DERANGEMENTS) for n in (80, 150, 250)),
    *(("total-mappings", n, _TOTAL_MAPPINGS) for n in (80, 160, 240)),
    *(("no-isolated-digraph", n, _NO_ISOLATED_DIGRAPH) for n in (75, 150, 225)),
    *(("2-matchings", n, _TWO_MATCHINGS) for n in (20, 30, 40)),
    *(("4-matchings", n, _FOUR_MATCHINGS) for n in (8, 12, 16)),
)

_CORE_SMOKE_CASES = tuple(_core_case(*entry) for entry in _CORE_SMOKE_ENTRIES)
_CORE_MAIN_CASES = tuple(_core_case(*entry) for entry in _CORE_MAIN_ENTRIES)
_CORE_EXHAUSTIVE_CASES = tuple(
    _core_case(*entry) for entry in (*_CORE_MAIN_ENTRIES, *_CORE_EXTRA_ENTRIES)
)


def _c2_cases() -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    direct_sizes = sorted({10, 15, 20, 25, 30, *range(10, 101, 10)})
    for n in direct_sizes:
        comparison_group = f"c2-vs-hand-3-regular/n{n}" if n % 10 == 0 else None
        cases.append(
            _case(
                key=f"c2/3-regular/n{n}",
                family="3-regular",
                category="c2",
                domain_size=n,
                builder=_c2_three_regular_problem,
                algorithm=AlgoName.INCREMENTAL3,
                comparison_group=comparison_group,
            )
        )
    for n in (8, 10, 12, 14, 16):
        cases.append(
            _case(
                key=f"c2/3-coloured-3-regular/n{n}",
                family="3-coloured-3-regular",
                category="c2",
                domain_size=n,
                builder=_c2_three_coloured_regular_problem,
                algorithm=AlgoName.INCREMENTAL3,
            )
        )
    for n in range(1, 21):
        cases.append(
            _case(
                key=f"c2/directed-3-in-3-out/n{n}",
                family="directed-3-in-3-out",
                category="c2",
                domain_size=n,
                builder=_c2_directed_three_in_three_out_problem,
                algorithm=AlgoName.INCREMENTAL3,
            )
        )
    for n in range(10, 101, 10):
        cases.append(
            _case(
                key=f"c2/3-regular-hand/n{n}",
                family="3-regular-hand",
                category="c2",
                domain_size=n,
                builder=partial(
                    _matrix_problem,
                    definition=_THREE_REGULAR,
                    constraint_factory=lambda domain: (
                        ("F", Comparator.EQ, 3 * domain),
                    ),
                ),
                algorithm=AlgoName.FASTV2,
                correction_divisor=6**n,
                comparison_group=f"c2-vs-hand-3-regular/n{n}",
            )
        )
    return tuple(cases)


_C2_CASES = _c2_cases()

_CARDINALITY_CASES = tuple(
    [
        _case(
            key=f"cardinality/3-regular-properly-4-coloured/n{n}",
            family="3-regular-properly-4-coloured",
            category="cardinality",
            domain_size=n,
            builder=_three_regular_four_coloured_problem,
            algorithm=AlgoName.FASTV2,
            correction_divisor=6**n,
        )
        for n in (10, 15, 20, 30, 50)
    ]
    + [
        _case(
            key=f"cardinality/directed-3-regular/n{n}",
            family="directed-3-regular",
            category="cardinality",
            domain_size=n,
            builder=_directed_three_regular_problem,
            algorithm=AlgoName.FASTV2,
            correction_divisor=36**n,
        )
        for n in range(10, 16)
    ]
)


_UNARY_FAMILIES: tuple[
    tuple[
        str,
        FormulaFactory,
        str,
        Callable[[int], UnaryBounds],
        Callable[[int], UnaryBounds],
    ],
    ...,
] = (
    ("row-column-Sx", _ROW_COLUMN, "Sx", _exact_half, _middle_interval),
    ("total-mappings-S", _TOTAL_MAPPINGS, "S", _exact_half, _middle_interval),
    (
        "no-isolated-digraph-S",
        _NO_ISOLATED_DIGRAPH,
        "S",
        _exact_half,
        _middle_interval,
    ),
    ("4-coloured-C1", _FOUR_COLOURED, "C1", _exact_quarter, _colour_interval),
)


def _unary_cases() -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    for family, definition, predicate, exact, interval in _UNARY_FAMILIES:
        for n in (20, 30, 40):
            for variant in ("unconstrained", "exact", "interval"):
                cases.append(
                    _case(
                        key=f"unary/{family}/{variant}/n{n}",
                        family=family,
                        category="unary-cardinality",
                        domain_size=n,
                        builder=partial(
                            _unary_problem,
                            definition=definition,
                            predicate=predicate,
                            variant=variant,
                            exact=exact,
                            interval=interval,
                        ),
                        algorithm=AlgoName.FASTV2,
                        variant=variant,
                    )
                )

    for family, definition, predicate, exact, interval in _UNARY_FAMILIES:
        if family == "no-isolated-digraph-S":
            continue
        for n in (100, 200, 300):
            for variant in ("exact", "interval"):
                cases.append(
                    _case(
                        key=f"unary-structure/{family}/{variant}/n{n}",
                        family=family,
                        category="unary-cardinality",
                        domain_size=n,
                        builder=partial(
                            _unary_problem,
                            definition=definition,
                            predicate=predicate,
                            variant=variant,
                            exact=exact,
                            interval=interval,
                        ),
                        algorithm=AlgoName.FASTV2,
                        variant=variant,
                        purposes=frozenset(("structure", "clique-gate")),
                    )
                )
    return tuple(cases)


_UNARY_CASES = _unary_cases()

_ALL_CASES = (
    *_CORE_SMOKE_CASES,
    *_CORE_EXHAUSTIVE_CASES,
    *_C2_CASES,
    *_CARDINALITY_CASES,
    *_UNARY_CASES,
)
_CASE_BY_KEY = {case.key: case for case in _ALL_CASES}
if len(_CASE_BY_KEY) != len(_ALL_CASES):
    raise RuntimeError("benchmark case keys must be unique")

_SUITES: dict[str, tuple[BenchmarkCase, ...]] = {
    "core-smoke": _CORE_SMOKE_CASES,
    "core-main": _CORE_MAIN_CASES,
    "core-exhaustive": _CORE_EXHAUSTIVE_CASES,
    "c2": _C2_CASES,
    "cardinality": _CARDINALITY_CASES,
    "unary": _UNARY_CASES,
    "all": _ALL_CASES,
}


def benchmark_suite_names() -> tuple[str, ...]:
    """Return suite names in stable display order."""

    return tuple(_SUITES)


def benchmark_cases(suite: str = "all") -> tuple[BenchmarkCase, ...]:
    """Return the concrete cases in ``suite`` without rebuilding the catalog."""

    try:
        return _SUITES[suite]
    except KeyError as error:
        choices = ", ".join(_SUITES)
        raise ValueError(f"unknown benchmark suite {suite!r}; choose one of: {choices}") from error


def benchmark_case(key: str) -> BenchmarkCase:
    """Return one concrete case by stable key."""

    try:
        return _CASE_BY_KEY[key]
    except KeyError as error:
        raise KeyError(f"unknown benchmark case: {key}") from error


__all__ = [
    "BenchmarkCase",
    "benchmark_case",
    "benchmark_cases",
    "benchmark_suite_names",
]
