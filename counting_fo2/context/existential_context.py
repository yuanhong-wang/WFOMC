from __future__ import annotations
import random

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generator
from logzero import logger
from itertools import product

from counting_fo2.cell_graph.components import Cell
from counting_fo2.fol.syntax import AtomicFormula, Pred, a, b
from counting_fo2.utils.multinomial import multinomial
from counting_fo2.utils.third_typing import RingElement


class BlockType(frozenset):
    pass


@dataclass(frozen=True)
class ExistentialTwoTable(object):
    code_ab: tuple[bool]
    code_ba: tuple[bool]
    preds: tuple[Pred]
    atoms: tuple[tuple[AtomicFormula]] = field(
        default=None, repr=False, init=False, hash=False, compare=False
    )

    def __post_init__(self):
        atoms = list()
        for p in self.preds:
            args_ab = [a, b]
            args_ba = [b, a]
            atoms.append((
                p(*args_ab),
                p(*args_ba)
            ))
        object.__setattr__(self, 'atoms', tuple(atoms))

    def get_evidences(self, reverse=False) -> frozenset[AtomicFormula]:
        evidences = set()
        for ab, ba, (atom_ab, atom_ba) in zip(self.code_ab, self.code_ba, self.atoms):
            if not reverse:
                evidences.add(atom_ab if ab else ~atom_ab)
                evidences.add(atom_ba if ba else ~atom_ba)
            else:
                evidences.add(atom_ab if ba else ~atom_ab)
                evidences.add(atom_ba if ab else ~atom_ba)
        return frozenset(evidences)

    def is_positive(self, pred: Pred) -> bool:
        idx = self.preds.index(pred)
        return self.code_ab[idx], self.code_ba[idx]

    def __str__(self):
        evidences = self.get_evidences()
        return str(evidences)

    def __repr__(self):
        return str(self)


class ExistentialContext(object):
    def __init__(self, cell_assignment: list[Cell],
                 ext_preds: list[Pred]):
        self.cell_assignment = cell_assignment
        self.cell_elements: dict[Cell, set[int]] = defaultdict(set)
        self.cell_config: dict[Cell, int] = defaultdict(lambda: 0)
        for idx, cell in enumerate(cell_assignment):
            self.cell_elements[cell].add(idx)
            self.cell_config[cell] += 1
        self.cells = list(self.cell_config.keys())
        self.ext_preds = ext_preds

        self.cb_config: dict[tuple[Cell, BlockType],
                             int] = defaultdict(lambda: 0)
        self.cb_elements: dict[
            tuple[Cell, BlockType], set[int]
        ] = defaultdict(set)
        # initialize cb_config and cb_elements
        for cell, num in self.cell_config.items():
            block_type = BlockType()
            for ext in self.ext_preds:
                if not cell.is_positive(ext):
                    block_type |= {ext}
            key = (cell, block_type)
            self.cb_config[key] = num
            self.cb_elements[key].update(self.cell_elements[cell])
        logger.debug('initial cell-block config: %s', self.cb_elements)

    def all_satisfied(self) -> bool:
        return all(self.cb_config[(cell, BlockType())] == self.cell_config[cell]
                   for cell in self.cells)

    def select_cell_block_type(self) -> tuple[Cell, BlockType]:
        assert not self.all_satisfied()
        for (cell, block), num in self.cb_config.items():
            if len(block) != 0 and num > 0:
                return cell, block

    def reduce_element(self, cell: Cell, block: BlockType) -> int:
        self.cell_config[cell] -= 1
        self.cb_config[(cell, block)] -= 1
        element = self.cb_elements[(cell, block)].pop()
        return element

    def reduce_block_type(self, block: BlockType,
                          etable: ExistentialTwoTable,
                          ab_or_ba: int = 0) -> BlockType:
        return block.difference(set(
            ext_pred for ext_pred in block
            if etable.is_positive(ext_pred)[ab_or_ba]
        ))

    def satisfied(self, block: BlockType,
                  overall_etable_config: dict[BlockType, int]) -> bool:
        if len(block) == 0:
            return True
        for etable, num in overall_etable_config.items():
            if num > 0:
                # ab_p
                block = self.reduce_block_type(block, etable)
            if len(block) == 0:
                return True
        return len(block) == 0

    def iter_etable_config(
        self,
        etable_weights: dict[Cell, dict[BlockType, RingElement]]
    ) -> Generator[dict[tuple[Cell, BlockType], dict[ExistentialTwoTable, int]]]:
        for raw_config in product(
                *list(
                    multinomial(len(etable_weights[cell]), num)
                    for (cell, _), num in self.cb_config.items()
                )
        ):
            etable_config = defaultdict(dict)
            for i, (cell, block) in enumerate(self.cb_config.keys()):
                config = raw_config[i]
                for j, etable in enumerate(etable_weights[cell].keys()):
                    etable_config[(cell, block)][etable] = config[j]
            yield etable_config

    def reduce_cb_config(self, etable_config:
                         dict[tuple[Cell, BlockType], tuple[int, ...]]) \
            -> frozenset[tuple[Cell, BlockType, int]]:
        reduced_cb_config = deepcopy(self.cb_config)
        for (cell, block), config in etable_config.items():
            for etable, num in config.items():
                reduced_block = self.reduce_block_type(
                    block, etable, ab_or_ba=1)
                reduced_cb_config[(cell, block)] -= num
                reduced_cb_config[(cell, reduced_block)] += num
        return frozenset(
            (*k, v) for k, v in reduced_cb_config.items() if v > 0
        )

    def sample_and_update(self, etable_config:
                          dict[tuple[Cell, frozenset[Pred]], tuple[int, ...]]) \
            -> dict[ExistentialTwoTable, list[int]]:
        etable_to_elements: dict[ExistentialTwoTable,
                                 list[int]] = defaultdict(list)
        updated_elements = dict()
        # sample
        for cb_type, config in etable_config.items():
            idx = 0
            elements = list(self.cb_elements[cb_type])
            # NOTE: we need to shuffle it again!
            random.shuffle(elements)
            updated_elements[cb_type] = defaultdict(list)
            for etable, num in config.items():
                etable_to_elements[etable] += elements[idx:(idx + num)]
                reduced_block = self.reduce_block_type(
                    cb_type[1], etable, ab_or_ba=1
                )
                sampled_elements = elements[idx:(idx + num)]
                updated_elements[cb_type][reduced_block].extend(
                    sampled_elements)
                idx += num

        # update
        for cb_type, config in updated_elements.items():
            for reduced_block, elements in config.items():
                self.cb_elements[cb_type].difference_update(elements)
                self.cb_elements[
                    (cb_type[0], reduced_block)
                ].update(elements)
                num = len(elements)
                self.cb_config[cb_type] -= num
                self.cb_config[(cb_type[0], reduced_block)] += num
        logger.debug('update eu_config: %s', self.cb_elements)
        return etable_to_elements

    def __str__(self):
        s = ''
        for (cell, block), num in self.cb_config.items():
            s += 'Cell {}, {}: {}\n'.format(cell, list(block), num)
        return s

    def __repr__(self):
        return str(self)
