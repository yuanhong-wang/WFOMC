from __future__ import annotations

from typing import List, Set, FrozenSet

from wfomc.fol.syntax import AtomicFormula, QuantifiedFormula, Const, Pred, Var
from wfomc.fol.utils import pad_vars


class MLN(object):
    def __init__(self, formulas: List[QuantifiedFormula],
                 weights: List[float] = None,
                 domain: Set[Const] = None,
                 predicate_definition: Set[Pred] = None,
                 unary_evidence: Set[AtomicFormula] = None):
        self.formulas: List[QuantifiedFormula] = formulas
        if weights is not None:
            if len(weights) != len(formulas):
                raise RuntimeError(
                    "Number of weights must match the number of formulas"
                )
        self.weights: List[float] = weights
        self.domain: Set[Const] = domain
        self.unary_evidence: Set[AtomicFormula] = unary_evidence

        # deal with predicate_definition
        preds = set()
        for formula in self.formulas:
            preds.update(formula.preds())
        if predicate_definition is not None:
            self.predicate_definition: FrozenSet[Pred] = frozenset(
                predicate_definition)
            # if predicate in formulas is not defined
            for pred in preds:
                if pred not in self.predicate_definition:
                    raise RuntimeError(
                        "Use the predicate %s without definition",
                        pred
                    )
            # if defined predicated is not in formulas,
            # add it with weight 0.0, i.e., exp(0) = 1
            for pred in self.predicate_definition:
                if pred not in preds:
                    vars = pad_vars(self.vars(), pred.arity)
                    self.formulas.append(
                        QuantifiedFormula.from_atom(pred(*vars)))
                    self.weights.append(0.0)
        else:
            self.predicate_definition = frozenset(preds)

        self.idx: int

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < self.size():
            ret = self.formulas[self.idx], self.weights[self.idx]
            self.idx += 1
            return ret
        else:
            raise StopIteration

    def preds(self) -> FrozenSet[Pred]:
        return self.predicate_definition

    def vars(self) -> FrozenSet[Var]:
        variables = set()
        for formula in self.formulas:
            variables.update(formula.vars())
        return frozenset(variables)

    def size(self) -> int:
        return len(self.formulas)

    def domain_size(self) -> int:
        return len(self.domain)

    def formula(self, index) -> QuantifiedFormula:
        return self.formulas[index]

    def weight(self, index) -> float:
        return self.weights[index]

    def is_hard(self, index) -> bool:
        return self.weight(index) == float('inf')

    def contain_existential_quantifier(self) -> bool:
        return any(formula.is_exist() for formula in self.formulas)

    def __str__(self):
        s = ''
        s += 'domain = {}\n'.format(','.join(
            str(element) for element in self.domain
        ))
        for f, w in self:
            s += '{} {}\n'.format(w, f)
        return s


class ComplexMLN(MLN):
    def __init__(self, formulas: List[QuantifiedFormula], weights: List[List[complex]] = None,
                 domain: Set[Const] = None, predicate_definition: Set[Pred] = None):
        super().__init__(formulas, weights, domain, predicate_definition)

    def __str__(self):
        s = ''
        s += 'domain = {}\n'.format(','.join(
            str(element) for element in self.domain
        ))
        for f, ws in self:
            s += '{} {}\n'.format(','.join(str(w) for w in ws), f)
        return s
