from __future__ import annotations

import functools

from functools import reduce
from typing import FrozenSet, List, Tuple
from dataclasses import dataclass, field
from logzero import logger
from sympy import Poly
from wfomc.cell_graph.utils import conditional_on

from wfomc.fol.syntax import AtomicFormula, Pred, Term, a, b, X
from wfomc.fol.utils import get_predicates
from wfomc.utils import Rational
from wfomc.utils.third_typing import RingElement


@dataclass(frozen=True)
class Cell(object):
    """
    In other words, the Unary types
    """
    code: Tuple[bool] = field(hash=False, compare=False)
    preds: Tuple[Pred] = field(hash=False, compare=False)
    # for hashing
    _identifier: FrozenSet[Tuple[Pred, bool]] = field(
        default=None, repr=False, init=False, hash=True, compare=True)

    def __post_init__(self):
        object.__setattr__(self, '_identifier',
                           frozenset(zip(self.preds, self.code)))

    @functools.lru_cache(maxsize=None)
    def get_evidences(self, term: Term) -> FrozenSet[AtomicFormula]:
        evidences: set[AtomicFormula] = set()
        for i, p in enumerate(self.preds):
            atom = p(*([term] * p.arity))
            if (self.code[i]):
                evidences.add(atom)
            else:
                evidences.add(~atom)
        return frozenset(evidences)

    @functools.lru_cache(maxsize=None)
    def is_positive(self, pred: Pred) -> bool:
        return self.code[self.preds.index(pred)]

    def negate(self, pred: Pred) -> Cell:
        idx = self.preds.index(pred)
        new_code = list(self.code)
        new_code[idx] = not new_code[idx]
        return Cell(tuple(new_code), self.preds)

    def drop_preds(self, preds: List[Pred] = None, prefixes: List[str] = None) -> Cell:
        if not preds and not prefixes:
            raise RuntimeError(
                'Dropped pred is not assigned'
            )
        if preds is not None:
            all_preds = [pred for pred in preds]
        else:
            all_preds = []

        if prefixes is not None:
            for prefix in prefixes:
                all_preds.extend(
                    get_predicates(prefix)
                )
        new_code, new_preds = zip(
            *[(c, p) for c, p in zip(self.code, self.preds)
              if p not in all_preds]
        )
        return Cell(tuple(new_code), tuple(new_preds))

    def __str__(self):
        evidences: frozenset[AtomicFormula] = self.get_evidences(X)
        # evidences = filter(lambda x: x.pred.name.startswith('skolem') or x.pred.name.startswith('aux') or x.pred.name == 'E', evidences)
        lits = [str(lit) for lit in evidences]
        lits.sort()
        return '^'.join(lits)

    def __repr__(self):
        return self.__str__()


class TwoTable(object):
    def __init__(self, models: dict[frozenset[AtomicFormula], RingElement],
                 gnd_lits: frozenset[AtomicFormula]):
        self.models = models
        self.gnd_lits = gnd_lits

    def get_weight(self, evidence: FrozenSet[AtomicFormula] = None) -> Poly:
        if not self.satisfiable(evidence):
            return Rational(0, 1)
        conditional_models = conditional_on(self.models, self.gnd_lits, evidence)
        ret = reduce(
            lambda a, b: a + b,
            conditional_models.values(),
            Rational(0, 1)
        )
        return ret

    def get_two_tables(self, evidence: FrozenSet[AtomicFormula] = None) \
            -> Tuple[FrozenSet[AtomicFormula], Poly]:
        if not self.satisfiable(evidence):
            return tuple()
        conditional_models = conditional_on(self.models, self.gnd_lits, evidence)
        return tuple(conditional_models.items())

    def satisfiable(self, evidence: FrozenSet[AtomicFormula] = None) -> bool:
        conditional_models = conditional_on(self.models, self.gnd_lits, evidence)
        if len(conditional_models) == 0:
            return False
        return True
