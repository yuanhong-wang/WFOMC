"""Command-line entry point for the new framework."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from wfomc.algo.core import (
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    algo_spec,
)
from wfomc.engine import solve
from wfomc.engine.runtime import RuntimeOptions
from wfomc.errors import WFOMCError
from wfomc.fol.grounding import LinearOrderEncoding
from wfomc.options import (
    BOUNDARY_PROFILE_PLANNER_STRATEGIES,
    BoundaryProfileOptions,
    EvidenceStrategy,
    ExistentialStrategy,
    WeightOptions,
)
from wfomc.result import WFOMCResult


@dataclass(frozen=True)
class CliResult:
    """Structured result returned by the programmatic CLI facade."""

    # Exact public result produced by the selected algorithm.
    result: WFOMCResult


_CLI_EVIDENCE_STRATEGIES = (
    EvidenceStrategy.CCS,
    EvidenceStrategy.LIFTED_PROFILES,
    EvidenceStrategy.GROUND_UNITS,
)
_PROPOSITIONAL_ALGOS = (
    AlgoName.PROPOSITIONAL,
    AlgoName.PROPOSITIONAL_REDUCED,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact WFOMC solver.")
    common = parser.add_argument_group("Common options")
    incremental3 = parser.add_argument_group("Incremental3-only options")
    propositional = parser.add_argument_group("Propositional-only options")
    boundary_profile = parser.add_argument_group(
        "Boundary-Profile-only options"
    )

    common.add_argument("--input", "-i", help="Input model path.")
    common.add_argument(
        "--algo",
        default=AlgoName.STANDARD.value,
        choices=tuple(
            algo.value
            for algo in AlgoName
            if algo_spec(algo).maturity in (AlgoMaturity.STABLE, AlgoMaturity.BETA)
        ),
        help="WFOMC algorithm name.",
    )
    common.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase diagnostic logging verbosity; repeat for DEBUG output.",
    )
    common.add_argument(
        "-e",
        "--evidence-strategy",
        choices=tuple(strategy.value for strategy in _CLI_EVIDENCE_STRATEGIES),
        help=(
            "Unary-evidence preparation strategy. Supported choices depend "
            "on the selected algorithm; omit it to use the algorithm default."
        ),
    )
    common.add_argument(
        "--exact-symbolic-backend",
        choices=("auto", "fmpq_mpoly", "fmpq_poly"),
        default="auto",
        help=(
            "Exact symbolic arithmetic backend. auto selects scalar, "
            "univariate, or multivariate arithmetic from the symbol count; "
            "fmpq_poly requires exactly one symbolic variable."
        ),
    )
    incremental3.add_argument(
        "--existential-strategy",
        choices=tuple(strategy.value for strategy in ExistentialStrategy),
        help=(
            "Incremental3 existential reduction strategy; only valid for "
            "incremental3, which defaults to counting."
        ),
    )
    propositional.add_argument(
        "-l",
        "--linear-order-encoding",
        choices=tuple(encoding.value for encoding in LinearOrderEncoding),
        help=(
            "Propositional order encoding. Only valid for propositional and "
            "propositional-reduced; defaults to pin when symmetry permits."
        ),
    )
    propositional.add_argument(
        "--ganak-path",
        help=(
            "Explicit Ganak executable path. Only valid for propositional "
            "and propositional-reduced."
        ),
    )
    boundary_profile.add_argument(
        "--bp-tree-reference-domain-size",
        type=int,
        help=(
            "Select one reusable Boundary-Profile tree using this reference "
            "domain size; omit it for domain-independent structural planning."
        ),
    )
    boundary_profile.add_argument(
        "--bp-planner-strategy",
        choices=BOUNDARY_PROFILE_PLANNER_STRATEGIES,
        default="auto",
        help=(
            "Boundary-Profile decomposition candidate policy. auto is the "
            "production default; the other choices support controlled "
            "planner experiments."
        ),
    )
    return parser


def run(
    input_path: str | Path,
    algo: AlgoName | str,
    *,
    evidence_strategy: EvidenceStrategy | str | None = None,
    existential_strategy: ExistentialStrategy | str | None = None,
    linear_order_encoding: LinearOrderEncoding | str | None = None,
    ganak_path: str | None = None,
    exact_symbolic_backend: str = "auto",
    bp_tree_reference_domain_size: int | None = None,
    bp_planner_strategy: str = "auto",
) -> CliResult:
    selected_algo = algo if isinstance(algo, AlgoName) else AlgoName(algo)
    if (
        linear_order_encoding is not None or ganak_path is not None
    ) and selected_algo not in _PROPOSITIONAL_ALGOS:
        invalid_options = []
        if linear_order_encoding is not None:
            invalid_options.append("--linear-order-encoding")
        if ganak_path is not None:
            invalid_options.append("--ganak-path")
        verb = "is" if len(invalid_options) == 1 else "are"
        raise ValueError(
            f"{' and '.join(invalid_options)} {verb} only valid for "
            "propositional algorithms"
        )
    if (
        bp_tree_reference_domain_size is not None
        or bp_planner_strategy != "auto"
    ) and selected_algo is not AlgoName.BOUNDARY_PROFILE:
        raise ValueError(
            "Boundary-Profile planning options are only valid for "
            "boundary-profile"
        )

    from wfomc.parser import parse_problem_file

    parsed_problem = parse_problem_file(input_path)
    selected_evidence_strategy = (
        evidence_strategy
        if isinstance(evidence_strategy, EvidenceStrategy)
        else EvidenceStrategy(evidence_strategy)
        if evidence_strategy is not None
        else None
    )
    selected_existential_strategy = (
        existential_strategy
        if isinstance(existential_strategy, ExistentialStrategy)
        else ExistentialStrategy(existential_strategy)
        if existential_strategy is not None
        else None
    )
    selected_linear_order_encoding = (
        linear_order_encoding
        if isinstance(linear_order_encoding, LinearOrderEncoding)
        else LinearOrderEncoding(linear_order_encoding)
        if linear_order_encoding is not None
        else None
    )
    result = solve(
        parsed_problem,
        algo=selected_algo,
        options=AlgoOptions(
            evidence_strategy=selected_evidence_strategy,
            existential_strategy=selected_existential_strategy,
            linear_order_encoding=selected_linear_order_encoding,
            weight_options=WeightOptions(
                exact_symbolic_backend=exact_symbolic_backend,
            ),
            boundary_profile_options=BoundaryProfileOptions(
                tree_reference_domain_size=bp_tree_reference_domain_size,
                planner_strategy=bp_planner_strategy,
            ),
        ),
        runtime=RuntimeOptions(propositional_ganak_path=ganak_path),
    )
    return CliResult(result=result)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input is None:
        parser.error("--input is required unless only requesting --help")

    _configure_logging(args.verbose)
    try:
        cli_result = run(
            args.input,
            args.algo,
            evidence_strategy=args.evidence_strategy,
            existential_strategy=args.existential_strategy,
            linear_order_encoding=args.linear_order_encoding,
            ganak_path=args.ganak_path,
            exact_symbolic_backend=args.exact_symbolic_backend,
            bp_tree_reference_domain_size=args.bp_tree_reference_domain_size,
            bp_planner_strategy=args.bp_planner_strategy,
        )
    except (WFOMCError, OSError, ValueError) as exc:
        parser.exit(2, f"wfomc: error: {type(exc).__name__}: {exc}\n")

    print(f"WFOMC ({args.algo}): {cli_result.result}")
    return 0


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


__all__ = ["CliResult", "build_parser", "main", "run"]
