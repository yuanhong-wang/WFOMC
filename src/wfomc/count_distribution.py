from __future__ import annotations

from logzero import logger

from wfomc.problems import WFOMCProblem
from wfomc.algo import Algo, standard_wfomc, fast_wfomc, incremental_wfomc

from wfomc.utils import MultinomialCoefficients, Rational
from wfomc.context import WFOMCContext
from wfomc.fol.syntax import Pred
from wfomc.utils.polynomial import coeff_dict, create_vars, expand


def count_distribution(problem: WFOMCProblem, preds: list[Pred],
                       algo: Algo = Algo.STANDARD) \
        -> dict[tuple[int, ...], Rational]:
    context = WFOMCContext(problem)
    # both standard and fast WFOMCs need precomputation
    if algo == Algo.STANDARD or algo == Algo.FAST or \
            algo == algo.FASTv2:
        MultinomialCoefficients.setup(len(problem.domain))
    leq_pred = Pred('LEQ', 2)
    if leq_pred in context.formula.preds():
        logger.info('Linear order axiom with the predicate LEQ is found')
        logger.info('Invoke incremental WFOMC')
        algo = Algo.INCREMENTAL
    else:
        leq_pred = None

    pred2weight = {}
    pred2sym = {}
    syms = create_vars('x0:{}'.format(len(preds)))
    for sym, pred in zip(syms, preds):
        if pred in pred2weight:
            continue
        weight = context.get_weight(pred)
        pred2weight[pred] = (weight[0] * sym, weight[1])
        pred2sym[pred] = sym
    context.weights.update(pred2weight)

    if algo == Algo.STANDARD:
        res = standard_wfomc(
            context.formula, context.domain, context.get_weight
        )
    elif algo == Algo.FAST:
        res = fast_wfomc(
            context.formula, context.domain, context.get_weight
        )
    elif algo == Algo.FASTv2:
        res = fast_wfomc(
            context.formula, context.domain, context.get_weight, True
        )
    elif algo == Algo.INCREMENTAL:
        res = incremental_wfomc(
            context.formula, context.domain,
            context.get_weight, leq_pred
        )

    symbols = [pred2sym[pred] for pred in preds]
    count_dist = {}
    res = expand(res)
    for degrees, coef in coeff_dict(res, symbols):
        count_dist[degrees] = coef
    return count_dist
