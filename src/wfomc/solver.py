import argparse
import os
import sys
from typing import Optional, Union

from contexttimer import Timer
from loguru import logger

from wfomc.algo import (
    Algo,
    LinearOrderEncoding,
    fast_wfomc,
    incremental_wfomc,
    incremental_wfomc3,
    propositional_wfomc,
    recursive_wfomc,
    resolve_linear_order_encoding,
    standard_wfomc,
)
from wfomc.context import IncrementalWFOMC3Context, UnaryEvidenceStrategy, WFOMCContext
from wfomc.fol import Counting, QuantifiedFormula
from wfomc.parser import parse_input
from wfomc.problems import WFOMCProblem
from wfomc.result import WFOMCResult
from wfomc.utils import MultinomialCoefficients, round_rational

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def resolve_unary_evidence_strategy(
    algo: Algo,
    requested: UnaryEvidenceStrategy,
    linear_order_encoding: Optional[Union[LinearOrderEncoding, str]] = None,
    has_linear_order: bool = False,
) -> UnaryEvidenceStrategy:
    """Resolve AUTO to the best supported unary evidence path."""
    if requested == UnaryEvidenceStrategy.CCS:
        return UnaryEvidenceStrategy.CCS
    if requested != UnaryEvidenceStrategy.AUTO:
        raise ValueError(f"Unsupported unary evidence strategy: {requested}")
    if algo in {Algo.FAST, Algo.RECURSIVE}:
        return UnaryEvidenceStrategy.CCS
    if algo == Algo.PROPOSITIONAL and has_linear_order:
        effective_loe = resolve_linear_order_encoding(linear_order_encoding)
        if effective_loe == LinearOrderEncoding.PIN:
            return UnaryEvidenceStrategy.CCS
    return UnaryEvidenceStrategy.AUTO


def _counting_formula_kind_and_comparator(formula: QuantifiedFormula) -> tuple[str, str | None]:
    if isinstance(formula.quantified_formula, QuantifiedFormula):
        scope = formula.quantified_formula.quantifier_scope
        kind = "binary"
    else:
        scope = formula.quantifier_scope
        kind = "unary"
    comparator = scope.comparator if isinstance(scope, Counting) else None
    return kind, comparator


def _validate_counting_quantifiers(problem: WFOMCProblem, algo: Algo) -> None:
    if algo == Algo.INCREMENTAL3:
        supported = {
            "unary": {"=", "<=", "mod"},
            "binary": {"=", "<=", "mod"},
        }
    else:
        supported = {
            "unary": {"=", "!=", "<", ">", "<=", ">="},
            "binary": {"="},
        }

    for formula in problem.sentence.cnt_formulas:
        kind, comparator = _counting_formula_kind_and_comparator(formula)
        if comparator is None or comparator in supported[kind]:
            continue
        allowed = ", ".join(sorted(supported[kind]))
        raise RuntimeError(
            f"{kind.capitalize()} counting comparator '{comparator}' is not "
            f"supported by the {algo} algorithm. Supported comparators: {allowed}."
        )


def wfomc(
    problem: WFOMCProblem,
    algo: Algo = Algo.STANDARD,
    unary_evidence_strategy: UnaryEvidenceStrategy = UnaryEvidenceStrategy.AUTO,
    linear_order_encoding: Optional[Union[LinearOrderEncoding, str]] = None,
    debug: bool = False,
) -> WFOMCResult:
    level = "DEBUG" if debug else "INFO"
    _handler_id = logger.add(
        sys.stderr, level=level, filter="wfomc", colorize=True, format=_LOG_FORMAT,
    )
    logger.enable("wfomc")
    try:
        MultinomialCoefficients.setup(len(problem.domain))

        if problem.contain_linear_order_axiom():
            logger.info('Linear order axiom with the predicate LEQ is found')
            if algo not in (
                Algo.INCREMENTAL,
                Algo.INCREMENTAL3,
                Algo.RECURSIVE,
                Algo.PROPOSITIONAL,
            ):
                raise RuntimeError("Linear order axiom is only supported by the "
                                   "incremental, incremental3, recursive, and "
                                   "propositional WFOMC algorithms")
        if problem.contain_predecessor_axiom():
            logger.info('Predecessor predicate PRED is found')
            if algo not in (Algo.INCREMENTAL, Algo.PROPOSITIONAL):
                raise RuntimeError("Predecessor axiom is only supported by the "
                                   "incremental and propositional WFOMC "
                                   "algorithms")

        effective_unary_evidence_strategy = resolve_unary_evidence_strategy(
            algo,
            unary_evidence_strategy,
            linear_order_encoding=linear_order_encoding,
            has_linear_order=problem.contain_linear_order_axiom(),
        )
        if problem.contain_unary_evidence():
            logger.info(
                'Unary evidence is found, requested strategy: {}, effective strategy: {}',
                unary_evidence_strategy,
                effective_unary_evidence_strategy,
            )

        _validate_counting_quantifiers(problem, algo)

        if problem.sentence.contain_modulo_counting_quantifier():
            logger.info('Modulo counting quantifier is found')
            if algo != Algo.INCREMENTAL3:
                raise RuntimeError("Modulo counting quantifier is only supported by the "
                                   "incremental WFOMC3 algorithm")

        logger.info(
            'Invoke WFOMC with {} algorithm and {} unary evidence strategy',
            algo,
            effective_unary_evidence_strategy,
        )
        if algo == Algo.INCREMENTAL3:
            context = IncrementalWFOMC3Context(problem, effective_unary_evidence_strategy)
        else:
            context = WFOMCContext(problem, effective_unary_evidence_strategy)

        with Timer() as t:
            if algo == Algo.STANDARD:
                res = standard_wfomc(context)
            elif algo == Algo.FAST:
                res = fast_wfomc(context)
            elif algo == Algo.FASTv2:
                res = fast_wfomc(context, True)
            elif algo == Algo.INCREMENTAL:
                res = incremental_wfomc(context, problem.circle_len)
            elif algo == Algo.RECURSIVE:
                res = recursive_wfomc(context)
            elif algo == Algo.PROPOSITIONAL:
                res = propositional_wfomc(
                    context,
                    linear_order_encoding=linear_order_encoding,
                )
            elif algo == Algo.INCREMENTAL3:
                res = incremental_wfomc3(context)

            if algo is not Algo.PROPOSITIONAL:
                res = context.decode_result(res)

        logger.info('WFOMC time: {}', t.elapsed)
        return WFOMCResult(res)
    finally:
        logger.remove(_handler_id)
        logger.disable("wfomc")


def parse_args():
    parser = argparse.ArgumentParser(
        description='WFOMC for MLN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='./check-points')
    parser.add_argument('--algo', '-a', type=Algo,
                        choices=list(Algo), default=Algo.FASTv2)
    parser.add_argument(
        '--unary-evidence-strategy',
        '-e',
        type=UnaryEvidenceStrategy,
        choices=list(UnaryEvidenceStrategy),
        default=UnaryEvidenceStrategy.AUTO,
    )
    parser.add_argument('--linear-order-encoding', '-l',
                        type=LinearOrderEncoding,
                        choices=list(LinearOrderEncoding),
                        default=None,
                        help='How the propositional counter encodes order '
                             'axioms (LEQ / PRED / CIRCULAR_PRED). Ignored by '
                             'the other algorithms. Default: pin.')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    level = "DEBUG" if args.debug else "INFO"

    try:
        logger.remove(0)
    except ValueError:
        pass

    logger.add(
        f'{args.output_dir}/log.txt',
        mode='w',
        level=level,
        filter="wfomc",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    )

    with Timer() as t:
        problem = parse_input(args.input, debug=args.debug)
    print(f'Parse input: {t.elapsed:.4f}s')

    res = wfomc(
        problem,
        algo=args.algo,
        unary_evidence_strategy=args.unary_evidence_strategy,
        linear_order_encoding=args.linear_order_encoding,
        debug=args.debug,
    )

    print(f'WFOMC (arbitrary precision): {res}')
    const_res = res.constant_value()
    if const_res is not None:
        round_val = round_rational(const_res)
        print(f'WFOMC (round): {round_val} (exp({round_val.ln()}))')
