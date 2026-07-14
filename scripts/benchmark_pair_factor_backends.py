#!/usr/bin/env python3
"""Compare batched exact PairFactor construction with PySDD and Ganak.

The benchmark captures real pair CNFs produced while solving three model files
and generates a typed FO2 family with a requested number of actual cells. Both
backends then compute the same map from
``(condition signature, projected relation mask)`` to an exact arithmetic
weight:

* PySDD compiles the CNF in process and evaluates its SDD once with the
  cell-graph factor algebra.
* Ganak performs one exact polynomial WMC.  Fresh polynomial variables mark
  condition/projection bits, so polynomial coefficients form the same factor
  map without invoking Ganak once per signature.

Potentially long solver calls are bounded by ``--timeout``.  The Ganak process
startup, DIMACS serialization, output parsing, and polynomial coefficient
extraction are included in its end-to-end timing.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import statistics
from dataclasses import dataclass
from math import ceil, log2
from pathlib import Path
from time import perf_counter
from typing import Any

from flint import fmpq, fmpq_mpoly, fmpq_mpoly_ctx

from wfomc.ganak import find_ganak, ganak_count
from wfomc.arithmetic import ArithmeticBackend, ArithmeticContext
from wfomc.cell_graph.enumerate_cells import enumerate_cells
from wfomc.cli import run
from wfomc.fol import FOLContext
from wfomc.fol.cnf import TseitinCNF, encode_tseitin
from wfomc.fol.grounding import ground_on_tuple


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelCase:
    label: str
    path: Path
    algo: str


@dataclass(frozen=True)
class PairWorkload:
    label: str
    cnf: TseitinCNF
    cells: tuple[Any, ...]
    predicate_weights: dict[Any, tuple[Any, Any]]
    arithmetic: ArithmeticContext
    projected_binary_preds: tuple[Any, ...]


@dataclass(frozen=True)
class PreparedWorkload:
    label: str
    cnf: TseitinCNF
    cell_count: int
    arithmetic: ArithmeticContext
    literal_weights: tuple[tuple[Any, Any], ...]
    projection: dict[int, int]
    free_factor: dict[int, Any]
    condition_signatures: int

    @property
    def marker_bits(self) -> int:
        return max(self.projection.values(), default=-1) + 1


@dataclass(frozen=True)
class BackendTiming:
    prepare_ms: float
    build_or_count_ms: float
    evaluate_or_extract_ms: float
    total_ms: float


@dataclass(frozen=True)
class BenchmarkRow:
    workload: PreparedWorkload
    factor_entries: int
    pysdd: BackendTiming
    ganak: BackendTiming


CASES = (
    ModelCase(
        "books",
        ROOT / "models/linear_order/unary_evidence/books-arragement.wfomcs",
        "incremental3",
    ),
    ModelCase(
        "predecessor",
        ROOT / "models/linear_order/predk/predecessor.wfomcs",
        "incremental",
    ),
    ModelCase(
        "markov3",
        ROOT / "models/linear_order/predk/3Markov_chain.mln",
        "incremental",
    ),
)


def _pair_module():
    return importlib.import_module("wfomc.cell_graph.compute_pair_factors")


def capture_real_workloads() -> list[PairWorkload]:
    """Capture every base/predecessor pair CNF built by the selected models."""

    build_module = importlib.import_module("wfomc.cell_graph.build")
    original = build_module.compute_pair_factors
    captured: list[PairWorkload] = []
    active_label = [""]
    occurrence: dict[str, int] = {}

    def capture(cnf, cells, predicate_weights, arithmetic, **kwargs):
        label = active_label[0]
        occurrence[label] = occurrence.get(label, 0) + 1
        suffix = occurrence[label]
        captured.append(
            PairWorkload(
                label=f"{label}/pair-{suffix}",
                cnf=cnf,
                cells=tuple(cells),
                predicate_weights=dict(predicate_weights),
                arithmetic=arithmetic,
                projected_binary_preds=tuple(
                    kwargs.get("projected_binary_preds", ())
                ),
            )
        )
        return original(cnf, cells, predicate_weights, arithmetic, **kwargs)

    build_module.compute_pair_factors = capture
    try:
        for case in CASES:
            active_label[0] = case.label
            run(case.path, case.algo)
    finally:
        build_module.compute_pair_factors = original
    return captured


def prepare_workload(workload: PairWorkload) -> PreparedWorkload:
    pair = _pair_module()
    condition_atoms = pair._condition_atoms(
        workload.cnf,
        workload.cells[0].preds,
    )
    condition_masks = {
        pair._condition_mask(condition_atoms, left, right)
        for left in workload.cells
        for right in workload.cells
    }
    counting_bits = 2 * len(workload.projected_binary_preds)
    projection = pair._counting_projection(
        workload.cnf,
        workload.projected_binary_preds,
    )
    projection.update(
        {
            workload.cnf.atom_to_var[atom]: counting_bits + index
            for index, atom in enumerate(condition_atoms)
        }
    )
    return PreparedWorkload(
        label=workload.label,
        cnf=workload.cnf,
        cell_count=len(workload.cells),
        arithmetic=workload.arithmetic,
        literal_weights=pair._literal_weights(
            workload.cnf,
            condition_atoms,
            workload.predicate_weights,
            workload.arithmetic,
        ),
        projection=projection,
        free_factor=pair._free_offdiagonal_factor(
            workload.cnf,
            workload.cells[0].preds,
            workload.projected_binary_preds,
            workload.predicate_weights,
            workload.arithmetic,
        ),
        condition_signatures=len(condition_masks),
    )


def _code_less_than(context: FOLContext, bits: tuple[Any, ...], limit: int):
    """Return a compact MSB-first Boolean formula for ``bits < limit``."""

    capacity = 1 << len(bits)
    if limit >= capacity:
        return context.true()
    bound = tuple(
        bool(limit & (1 << shift)) for shift in reversed(range(len(bits)))
    )
    equal_prefix: list[Any] = []
    lower_cases: list[Any] = []
    for bit, bound_bit in zip(bits, bound):
        if bound_bit:
            lower_cases.append(context.conjunction(*equal_prefix, ~bit))
        equal_prefix.append(bit if bound_bit else ~bit)
    return context.disjunction(*lower_cases)


def build_cell_scaling_workload(cell_count: int) -> PairWorkload:
    """Build a real FO2 family with exactly ``cell_count`` one-types.

    Each type bit has a unary predicate ``P_i`` and a binary predicate ``R_i``
    constrained by ``P_i(X) <-> R_i(X,Y)``. The diagonal grounding therefore
    has one valid cell per unary bit code, while the pair grounding depends on
    both the left and right codes. A compact ``code < cell_count`` formula
    trims the final power-of-two family when, for example, 200 cells are
    requested.
    """

    if cell_count < 2:
        raise ValueError("cell-scaling workload requires at least two cells")
    bit_count = ceil(log2(cell_count))
    context = FOLContext()
    x, y = context.vars("X Y")
    a, b, c = context.constants("a b c")
    unary_preds = tuple(
        context.predicate(f"CellBit{index}", 1) for index in range(bit_count)
    )
    binary_preds = tuple(
        context.predicate(f"CellLink{index}", 2) for index in range(bit_count)
    )

    unary_bits = tuple(predicate(x) for predicate in unary_preds)
    formula = context.conjunction(
        _code_less_than(context, unary_bits, cell_count),
        *(
            context.iff(unary(x), binary_pred(x, y))
            for unary, binary_pred in zip(unary_preds, binary_preds)
        ),
    )
    predicate_order = tuple(
        sorted(
            unary_preds + binary_preds,
            key=lambda predicate: predicate.cache_key_parts(),
        )
    )
    cell_cnf = encode_tseitin(ground_on_tuple(formula, c))
    cells = enumerate_cells(cell_cnf, predicate_order)
    if len(cells) != cell_count:
        raise AssertionError(
            f"requested {cell_count} cells but constructed {len(cells)}"
        )

    pair_formula = ground_on_tuple(formula, a, b) & ground_on_tuple(
        formula,
        b,
        a,
    )
    pair_cnf = encode_tseitin(pair_formula)
    arithmetic = ArithmeticContext(ArithmeticBackend.FMPQ)
    one = arithmetic.one()
    weights = {predicate: (one, one) for predicate in unary_preds}
    weights.update(
        {
            predicate: (
                arithmetic.from_int(index + 2),
                arithmetic.from_int(index + 3),
            )
            for index, predicate in enumerate(binary_preds)
        }
    )
    return PairWorkload(
        label=f"cell-scale/{cell_count}",
        cnf=pair_cnf,
        cells=cells,
        predicate_weights=weights,
        arithmetic=arithmetic,
        projected_binary_preds=(),
    )


def pysdd_factor(
    workload: PreparedWorkload,
) -> tuple[dict[int, Any], BackendTiming]:
    pair = _pair_module()
    total_started = perf_counter()
    prepare_started = perf_counter()
    # Inputs are already normalized; this timing represents only lightweight
    # backend setup so the row shape remains comparable to Ganak.
    weights = workload.literal_weights
    projection = workload.projection
    prepare_ms = (perf_counter() - prepare_started) * 1000

    build_started = perf_counter()
    circuit = pair._SddCircuit(workload.cnf, workload.arithmetic)
    build_ms = (perf_counter() - build_started) * 1000
    try:
        evaluate_started = perf_counter()
        factor = circuit.evaluate(weights, projection)
        factor = pair._factor_multiply(
            factor,
            workload.free_factor,
            workload.arithmetic,
        )
        evaluate_ms = (perf_counter() - evaluate_started) * 1000
    finally:
        circuit.close()
    return factor, BackendTiming(
        prepare_ms=prepare_ms,
        build_or_count_ms=build_ms,
        evaluate_or_extract_ms=evaluate_ms,
        total_ms=(perf_counter() - total_started) * 1000,
    )


def _expanded_poly_context(
    workload: PreparedWorkload,
) -> tuple[fmpq_mpoly_ctx, int]:
    base_names = workload.arithmetic.symbolic_variables
    marker_names = tuple(
        f"__wfomc_pair_marker_{index}" for index in range(workload.marker_bits)
    )
    return fmpq_mpoly_ctx.get(base_names + marker_names, "lex"), len(base_names)


def _as_expanded_poly(value: Any, context: fmpq_mpoly_ctx) -> fmpq_mpoly:
    if isinstance(value, fmpq_mpoly):
        return value.project_to_context(context)
    return context.constant(value)


def _extract_marker_factor(
    polynomial: fmpq_mpoly,
    workload: PreparedWorkload,
    base_symbol_count: int,
) -> dict[int, Any]:
    arithmetic = workload.arithmetic
    base_names = arithmetic.symbolic_variables
    base_context = (
        fmpq_mpoly_ctx.get(base_names, "lex") if base_symbol_count else None
    )
    result: dict[int, Any] = {}
    for monomial, coefficient in zip(polynomial.monoms(), polynomial.coeffs()):
        marker_exponents = monomial[base_symbol_count:]
        if any(exponent not in (0, 1) for exponent in marker_exponents):
            raise AssertionError(
                f"non-multilinear Ganak marker monomial: {monomial}"
            )
        mask = sum(
            (1 << index)
            for index, exponent in enumerate(marker_exponents)
            if exponent
        )
        if base_context is None:
            value = arithmetic.coerce(fmpq(coefficient))
        else:
            value = base_context.from_dict(
                {tuple(monomial[:base_symbol_count]): coefficient}
            )
            value = arithmetic.coerce(value)
        result[mask] = result.get(mask, arithmetic.zero()) + value
    return {mask: value for mask, value in result.items() if not arithmetic.is_zero(value)}


def ganak_factor(
    workload: PreparedWorkload,
    *,
    ganak_path: str,
    timeout: float,
) -> tuple[dict[int, Any], BackendTiming]:
    pair = _pair_module()
    total_started = perf_counter()
    prepare_started = perf_counter()
    context, base_symbol_count = _expanded_poly_context(workload)
    weights: dict[int, tuple[Any, Any]] = {}
    for variable in range(1, workload.cnf.n_vars + 1):
        positive, negative = workload.literal_weights[variable]
        positive_poly = _as_expanded_poly(positive, context)
        negative_poly = _as_expanded_poly(negative, context)
        marker_bit = workload.projection.get(variable)
        if marker_bit is not None:
            positive_poly *= context.gen(base_symbol_count + marker_bit)
        weights[variable] = (positive_poly, negative_poly)
    prepare_ms = (perf_counter() - prepare_started) * 1000

    count_started = perf_counter()
    polynomial = ganak_count(
        workload.cnf.n_vars,
        workload.cnf.clauses,
        weights,
        symbolic=True,
        npolyvars=context.nvars(),
        poly_ctx=context,
        ganak_path=ganak_path,
        timeout=timeout,
    )
    count_ms = (perf_counter() - count_started) * 1000
    if not isinstance(polynomial, fmpq_mpoly):
        raise TypeError(f"expected Ganak polynomial, got {type(polynomial).__name__}")

    extract_started = perf_counter()
    factor = _extract_marker_factor(polynomial, workload, base_symbol_count)
    factor = pair._factor_multiply(
        factor,
        workload.free_factor,
        workload.arithmetic,
    )
    extract_ms = (perf_counter() - extract_started) * 1000
    return factor, BackendTiming(
        prepare_ms=prepare_ms,
        build_or_count_ms=count_ms,
        evaluate_or_extract_ms=extract_ms,
        total_ms=(perf_counter() - total_started) * 1000,
    )


def _median_timing(values: list[BackendTiming]) -> BackendTiming:
    return BackendTiming(
        prepare_ms=statistics.median(value.prepare_ms for value in values),
        build_or_count_ms=statistics.median(
            value.build_or_count_ms for value in values
        ),
        evaluate_or_extract_ms=statistics.median(
            value.evaluate_or_extract_ms for value in values
        ),
        total_ms=statistics.median(value.total_ms for value in values),
    )


def benchmark_one(
    workload: PreparedWorkload,
    *,
    ganak_path: str,
    timeout: float,
    trials: int,
) -> BenchmarkRow:
    pysdd_times: list[BackendTiming] = []
    ganak_times: list[BackendTiming] = []
    reference: dict[int, Any] | None = None

    for _ in range(trials):
        gc.collect()
        factor, timing = pysdd_factor(workload)
        if reference is None:
            reference = factor
        elif factor != reference:
            raise AssertionError(f"PySDD is nondeterministic for {workload.label}")
        pysdd_times.append(timing)

    for _ in range(trials):
        gc.collect()
        factor, timing = ganak_factor(
            workload,
            ganak_path=ganak_path,
            timeout=timeout,
        )
        if factor != reference:
            differing = sorted(set(factor) | set(reference or {}))
            preview = [
                (mask, (reference or {}).get(mask), factor.get(mask))
                for mask in differing
                if (reference or {}).get(mask) != factor.get(mask)
            ][:5]
            raise AssertionError(
                f"Ganak/PySDD mismatch for {workload.label}: {preview}"
            )
        ganak_times.append(timing)

    return BenchmarkRow(
        workload=workload,
        factor_entries=len(reference or {}),
        pysdd=_median_timing(pysdd_times),
        ganak=_median_timing(ganak_times),
    )


def print_rows(rows: list[BenchmarkRow]) -> None:
    print(
        "| workload | cells | pairs | vars | clauses | signatures | "
        "marker bits | entries | "
        "PySDD compile | PySDD eval | PySDD total | Ganak count | "
        "Ganak extract | Ganak total | PySDD/Ganak |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in rows:
        workload = row.workload
        speedup = row.pysdd.total_ms / row.ganak.total_ms
        print(
            f"| {workload.label} | {workload.cell_count} | "
            f"{workload.cell_count ** 2} | {workload.cnf.n_vars} | "
            f"{len(workload.cnf.clauses)} | {workload.condition_signatures} | "
            f"{workload.marker_bits} | {row.factor_entries} | "
            f"{row.pysdd.build_or_count_ms:.3f} ms | "
            f"{row.pysdd.evaluate_or_extract_ms:.3f} ms | "
            f"{row.pysdd.total_ms:.3f} ms | "
            f"{row.ganak.build_or_count_ms:.3f} ms | "
            f"{row.ganak.evaluate_or_extract_ms:.3f} ms | "
            f"{row.ganak.total_ms:.3f} ms | {speedup:.2f}x |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ganak", help="Ganak binary; defaults to GANAK/PATH")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--cell-counts",
        default="8,16,32,64,128,256",
        help="comma-separated real cell counts for the generated FO2 family",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="comma-separated workload-label substrings to retain",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    ganak_path = find_ganak(args.ganak)

    real = [prepare_workload(item) for item in capture_real_workloads()]
    cell_counts = [
        int(value) for value in args.cell_counts.split(",") if value.strip()
    ]
    workloads = real + [
        prepare_workload(build_cell_scaling_workload(count))
        for count in cell_counts
    ]
    filters = tuple(value for value in args.filter.split(",") if value)
    if filters:
        workloads = [
            workload
            for workload in workloads
            if any(value in workload.label for value in filters)
        ]

    print(f"ganak={ganak_path}")
    print(f"trials={args.trials} timeout={args.timeout}s")
    rows: list[BenchmarkRow] = []
    for workload in workloads:
        print(f"benchmarking {workload.label}...", flush=True)
        rows.append(
            benchmark_one(
                workload,
                ganak_path=ganak_path,
                timeout=args.timeout,
                trials=args.trials,
            )
        )
    print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
