from __future__ import annotations

from wfomc.fol.syntax import AtomicFormula
from wfomc.utils.third_typing import RingElement


def conditional_on(models: dict[frozenset[AtomicFormula], RingElement],
                   gnd_lits: frozenset[AtomicFormula],
                   evidence: frozenset[AtomicFormula] = None) \
        -> dict[frozenset[AtomicFormula], RingElement]:
    """
    :param models: a dictionary of models and their weights
    :param gnd_lits: all possible ground literals in the formula
    :param evidence: a set of ground literals
    :return: a dictionary of models and their weights conditioned on the evidence
    """
    if evidence is None:
        return models
    filtered_models = dict(filter(
        lambda x: evidence.intersection(gnd_lits).issubset(x[0]),
        models.items()
    ))
    return filtered_models
