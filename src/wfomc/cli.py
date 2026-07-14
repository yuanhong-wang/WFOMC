"""Command-line entry point for the new framework."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from wfomc.algo import (
    AlgoMaturity,
    AlgoName,
    AlgoOptions,
    ExistentialStrategy,
    algo_spec,
)
from wfomc.api import solve
from wfomc.errors import WFOMCError
from wfomc.result import WFOMCResult


@dataclass(frozen=True)
class CliResult:
    result: WFOMCResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact WFOMC solver.")
    parser.add_argument("--input", "-i", help="Input model path.")
    parser.add_argument(
        "--algo",
        default=AlgoName.STANDARD.value,
        choices=tuple(
            algo.value
            for algo in AlgoName
            if algo_spec(algo).maturity in (AlgoMaturity.STABLE, AlgoMaturity.BETA)
        ),
        help="WFOMC algorithm name.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase diagnostic logging verbosity; repeat for DEBUG output.",
    )
    parser.add_argument(
        "--existential-strategy",
        choices=tuple(strategy.value for strategy in ExistentialStrategy),
        help=(
            "Existential reduction strategy. incremental3 defaults to counting; "
            "other algorithms use skolem."
        ),
    )
    return parser


def run(
    input_path: str | Path,
    algo: AlgoName | str,
    *,
    existential_strategy: ExistentialStrategy | str | None = None,
) -> CliResult:
    from wfomc.parser import parse_problem_file

    selected_algo = algo if isinstance(algo, AlgoName) else AlgoName(algo)
    parsed_problem = parse_problem_file(input_path)
    selected_existential_strategy = (
        existential_strategy
        if isinstance(existential_strategy, ExistentialStrategy)
        else ExistentialStrategy(existential_strategy)
        if existential_strategy is not None
        else None
    )
    result = solve(
        parsed_problem,
        algo=selected_algo,
        options=AlgoOptions(existential_strategy=selected_existential_strategy),
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
            existential_strategy=args.existential_strategy,
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
