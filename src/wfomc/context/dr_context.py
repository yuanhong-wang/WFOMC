from __future__ import annotations
from copy import deepcopy
import hashlib
from collections import defaultdict
import math
import numpy as np
from logzero import logger
from wfomc.fol.sc2 import SC2
from wfomc.fol.utils import new_predicate, convert_counting_formula
from wfomc.fol.syntax import (
    AUXILIARY_PRED_NAME,
    X,
    Y,
    AtomicFormula,
    Const,
    Pred,
    QFFormula,
    top,
    a,
    b,
    Top,
)
from wfomc.network.constraint import CardinalityConstraint
from wfomc.fol.syntax import *
from wfomc.problems import WFOMCProblem
from wfomc.utils.polynomial_flint import RingElement, Rational
from itertools import product
from flint import fmpq, fmpq_poly


class DRWFOMCContext(object):

    def __init__(self, problem: WFOMCProblem):
        """
        Args:
            problem (WFOMCProblem): WFOMC问题实例
        """
        # 域、句子、权重和基数约束
        self.problem = deepcopy(problem)
        self.domain: set[Const] = problem.domain
        self.sentence: SC2 = problem.sentence
        self.weights: dict[Pred, tuple[Rational, Rational]] = problem.weights
        self.cardinality_constraint: CardinalityConstraint = (
            problem.cardinality_constraint
        )
        self.repeat_factor = 1

        logger.info("sentence: \n%s", self.sentence)
        logger.info("domain: \n%s", self.domain)
        logger.info("weights:")
        for pred, w in self.weights.items():
            logger.info("%s: %s", pred, w)
        logger.info("cardinality constraint: %s", self.cardinality_constraint)

        self.formula: QFFormula  # 无量词公式
        # 处理线性序公理
        if problem.contain_linear_order_axiom():
            self.leq_pred: Pred = Pred("LEQ", 2)
        else:
            self.leq_pred: Pred = None

        self.uni_formula: QuantifiedFormula = Top
        self.ext_preds: list[QuantifiedFormula] = []

        # 单双层
        # 每层是mod = <=
        # 单层，谓词是一元
        # 双层 谓词是二元
        # 计数
        self.cnt_preds: list[QuantifiedFormula] = []  # 计数谓词列表
        self.cnt_params: list[int] = []  # 计数参数 k (int)
        self.cnt_remainder: list[int] = []  # 余数 r (int)
        # unary
        self.mod_pred_index = []  # 模运算谓词索引
        self.exist_mod = False  # 是否存在模运算
        self.unary_mod_constraints = []  # 一元模约束 [(Pred, r, k), …]
        self.unary_eq_constraints: list[tuple] = []  # [(pred, k), ...]
        self.unary_le_constraints = []  # [(pred, k_max), ...]

        # <=
        self.exist_le = False  # "是否有<="
        self.le_pred = []  # 小于等于谓词列表
        self.le_index = []  # 小于等于谓词索引
        # 比较器处理函数映射
        self.comparator_handlers = {  # 比较器处理函数映射
            "mod": self._handle_mod,
            "=": self._handle_eq,
            "<=": self._handle_le,
        }

        self._build()

        self.c_type_shape = tuple()
        self.build_c_type_shape()

        self.binary_evidence = []
        self.get_binary_evidence()

        # binary cardinality_constraints is underconstruction 这部分采用的是symbolic weight
        self.card_preds = []
        self.card_ccs = []
        self.card_vars = []
        self.build_cardinality_constraints()
        self.update_repeat_factor() # 更新 repeat factor, 比如 m-odd 这个输入的例子 需要除以 domain size，也就是n choose 1。

    def update_repeat_factor(self):
        # 由于m-odd 这个输入的例子 需要除以 domain size
        if hasattr(self, 'unary_eq_constraints') and self.unary_eq_constraints:
            constraint_names = {constraint[0].name for constraint in self.unary_eq_constraints}
            if 'Odd' in constraint_names and 'U' in constraint_names:
                self.repeat_factor = len(self.problem.domain)
                print("change repeat factor to:", self.repeat_factor)

    def stop_condition(self, last_target_c):
        """


        Args:
            last_target_c: 

        Returns:
            bool: 
        """
        if self.exist_le:
            pred_state = last_target_c[1:]
            for idx in self.le_index:
                if idx < len(pred_state) and pred_state[idx] > 0:
                    return True
            for i in range(len(pred_state)):
                if i not in self.le_index and pred_state[i] != 0:
                    return False
            return True
        else:
            return all(i == 0 for i in last_target_c[1:])

    def _extract_formula(self, formula):
        if isinstance(formula.quantified_formula, QuantifiedFormula):
            inner_formula = formula.quantified_formula
            return (
                "binary",
                inner_formula.quantified_formula,
                inner_formula.quantifier_scope,
            )
        else:
            return (
                "unary",
                formula.quantified_formula,
                formula.quantifier_scope,
            )

    def _add_aux_equiv(self, inner_formula):
        aux_pred = new_predicate(2, AUXILIARY_PRED_NAME)
        self.uni_formula = self.uni_formula & inner_formula.equivalent(
            aux_pred(X, Y)
        )
        self.cnt_preds.append(aux_pred)
        return aux_pred

    def _handle_mod(
        self, type, idx, inner_formula, qscope, param, _
    ):

        r, k = param

        # unary mod
        # unary is under construction
        if (
            type == "unary"
            and isinstance(inner_formula, AtomicFormula)
            and inner_formula.pred.arity == 1
            and inner_formula.args == (qscope.quantified_var,)
        ):
            self.unary_mod_constraints.append(
                (inner_formula.pred, r, k)
            )
            return
        elif type == "binary":
            # binary mod
            self.exist_mod = True
            self.mod_pred_index.append(idx)
            self.cnt_remainder.append(r)
            self.cnt_params.append(k)
            self._add_aux_equiv(inner_formula)

    def _handle_eq(self, type, idx, inner_formula, qscope, param, comparator):
        """
        处理等号量词 ∃_{=m}
        Args:
            type: unary 还是 binary
            idx: 索引
            inner_formula: 内部公式
            qscope: 量词作用域
            param: 参数 k
            comparator: 比较器
        """
        # unary ∃_{=k} X  A(X)
        if (
            type == "unary"
            and isinstance(inner_formula, AtomicFormula)
            and inner_formula.pred.arity == 1
            and inner_formula.args == (qscope.quantified_var,)
        ):
            self.unary_eq_constraints.append(
                (inner_formula.pred, param)
            )
            return
        elif type == "binary":
            self.cnt_remainder.append(None)
            self.cnt_params.append(param)
            aux_pred = self._add_aux_equiv(inner_formula)

    def _handle_le(self, type, idx, inner_formula, qscope, param, comparator):

        if (
            type == "unary"
            and isinstance(inner_formula, AtomicFormula)
            and inner_formula.pred.arity == 1
            and inner_formula.args == (qscope.quantified_var,)
        ):
            self.unary_le_constraints.append(
                (inner_formula.pred, param)
            )
            return
        elif type == "binary":
            self.cnt_remainder.append(None)
            self.cnt_params.append(param)
            aux_pred = self._add_aux_equiv(inner_formula)
            self.le_pred.append(aux_pred)
            self.exist_le = True

    def _build(self):
        """
        预处理逻辑公式，将其转换为可处理的无量词形式，并引入辅助谓词
        """
        # 提取全称公式
        self.uni_formula = self.sentence.uni_formula
        while not isinstance(self.uni_formula, QFFormula):
            self.uni_formula = self.uni_formula.quantified_formula
        ext_formulas = self.sentence.ext_formulas  # 存在量词公式
        cnt_formulas = self.sentence.cnt_formulas  # 计数量词公式

        # 处理存在公式
        for (
            formula
        ) in (
            ext_formulas
        ):  # 这里我们在处理计数量词之前，处理存在量词，来实现UFO2。而把计数量词的处理和这部分分开，也就是对应于论文
            self.uni_formula = self.uni_formula & self._skolemize_one_formula(
                formula)

        # 处理计数公式
        for idx, formula in enumerate(cnt_formulas):
            type, inner_formula, qscope = self._extract_formula(
                formula
            )  # 因为可能是双层或单层，所以需要拆分
            comparator = qscope.comparator  # 'mod' / '=' / '<=' / ...
            cnt_param_raw = qscope.count_param  # (r,k) 或 int

            # 根据 comparator 分派到对应的 handler
            idx = len(self.cnt_preds)  # 用当前 cnt_preds 长度算下标
            # 也就是说，cnt_formulas 和 cnt_preds的长度是不同的。unary mod不会添加进cnt_preds中。为了跳过unary mod,不采用手动累加 idx = idx + 1。是因为idx 必须始终与 cnt_preds 的当前长度保持同步，保持新谓词下标依然连续、正确，
            self.comparator_handlers[comparator](
                type, idx, inner_formula, qscope, cnt_param_raw, comparator
            )

        self.all_preds = self.ext_preds + self.cnt_preds
        self._pred2idx = {
            pred: i for i, pred in enumerate(self.all_preds)
        }
        self.le_index = [
            self.all_preds.index(pred) for pred in self.le_pred
        ]

    def _skolemize_one_formula(self, formula: QuantifiedFormula) -> QFFormula:
        quantified_formula = formula.quantified_formula
        quantifier_num = 1
        while not isinstance(
            quantified_formula, QFFormula
        ):
            quantified_formula = quantified_formula.quantified_formula
            quantifier_num += 1
        skolem_formula: QFFormula = top
        # ext_formula 指的是存在量词内核的无量词公式部分，例如 f(X,Y)
        ext_formula = quantified_formula
        if not isinstance(
            ext_formula, AtomicFormula
        ):
            aux_pred = new_predicate(
                quantifier_num, AUXILIARY_PRED_NAME
            )
            aux_atom = (
                aux_pred(X, Y) if quantifier_num == 2 else aux_pred(X)
            )
            skolem_formula = skolem_formula & (
                ext_formula.equivalent(aux_atom)
            )
            ext_formula = (
                aux_atom
            )
        if quantifier_num == 2:
            skolem_pred = new_predicate(
                1, SKOLEM_PRED_NAME
            )
            skolem_atom = skolem_pred(X)
        elif quantifier_num == 1:
            skolem_pred = new_predicate(
                0, SKOLEM_PRED_NAME
            )
            skolem_atom = skolem_pred()

        skolem_formula = skolem_formula & (
            skolem_atom | ~ext_formula
        )
        self.weights[skolem_pred] = (
            Rational(1, 1),
            Rational(-1, 1),
        )
        return skolem_formula

    def build_c_type_shape(self):
        self.c_type_shape = list(
            2 for _ in self.ext_preds
        )
        for idx, k in enumerate(self.cnt_params):
            if idx in self.mod_pred_index:
                self.c_type_shape.append(k)
            else:
                self.c_type_shape.append(k + 1)

    def get_binary_evidence(self):
        ext_atoms = list(
            (
                (~pred(a, b), ~pred(b, a)),
                (~pred(a, b), pred(b, a)),
                (pred(a, b), ~pred(b, a)),
                (pred(a, b), pred(b, a)),
            )
            for pred in self.ext_preds[::-1]
        )
        cnt_atoms = list(
            (
                (~pred(a, b), ~pred(b, a)),
                (~pred(a, b), pred(b, a)),
                (pred(a, b), ~pred(b, a)),
                (pred(a, b), pred(b, a)),
            )
            for pred in self.cnt_preds[::-1]
        )
        for atoms in product(*cnt_atoms, *ext_atoms):
            self.binary_evidence.append(
                frozenset(sum(atoms, start=()))
            )

    def build_cardinality_constraints(self):
        if self.contain_cardinality_constraint():
            self.cardinality_constraint.build()
            self.weights.update(
                self.cardinality_constraint.transform_weighting(
                    self.get_weight,
                )
            )

    def decode_result(self, res: RingElement) -> Rational:
        if not self.contain_cardinality_constraint():
            res = res / self.repeat_factor
        else:
            res = self.cardinality_constraint.decode_poly(
                res) / self.repeat_factor
        if self.leq_pred is not None:
            res *= Rational(math.factorial(len(self.domain)), 1)
        return res

    def contain_cardinality_constraint(self) -> bool:

        return (
            self.cardinality_constraint is not None
            and not self.cardinality_constraint.empty()
        )

    def contain_linear_order_axiom(self) -> bool:

        return self.problem.contain_linear_order_axiom()

    def build_t_updates(self, r, n_cells, domain_size):

        t_updates = defaultdict(
            lambda: defaultdict(lambda: Rational(0, 1))
        )

        if self.exist_mod:
            final_list = [
                tuple(range(2)) for _ in self.ext_preds
            ]
            for idx, k in enumerate(self.cnt_params):
                if idx in self.mod_pred_index:
                    final_list += [tuple(range(k))]
                else:
                    final_list += [tuple(range(k + 1))]

            all_ts = list(product(*(final_list)))

        else:
            all_ts = list(
                product(
                    *(
                        [tuple(range(2)) for _ in self.ext_preds]
                        + [tuple(range(i + 1)) for i in self.cnt_params]
                    )
                )
            )
        for i in range(n_cells):
            for j in range(n_cells):
                for (
                    t1
                ) in (
                    all_ts
                ):
                    for t2 in all_ts:
                        for (dt, reverse_dt), rijt in r[(i, j)].items():
                            t1_new = [a - b for a, b in zip(t1, dt)]
                            t2_new = [a - b for a, b in zip(t2, reverse_dt)]
                            if self.exist_mod:
                                for p, k_i in enumerate(
                                    self.cnt_params
                                ):
                                    index = (
                                        len(self.ext_preds) + p
                                    )
                                    if p in self.mod_pred_index:
                                        t1_new[index] %= k_i
                                        t2_new[index] %= k_i

                            if any(
                                t1_new[len(self.ext_preds) + p] < 0
                                or t2_new[len(self.ext_preds) + p] < 0
                                for p in range(len(self.cnt_params))
                            ):
                                continue
                            for idx in range(
                                len(self.ext_preds)
                            ):
                                t1_new[idx] = max(t1_new[idx], 0)
                                t2_new[idx] = max(t2_new[idx], 0)
                            c1 = (i,) + t1
                            c2 = (j,) + t2
                            c1_new = (i,) + tuple(t1_new)
                            c2_new = (j,) + tuple(t2_new)
                            t_updates[(c1, c2)][
                                (c1_new, c2_new)
                            ] += rijt
        return t_updates

    def build_weight(self, cells, cell_graph):

        n_cells = len(cells)
        w2t = dict()
        w = defaultdict(
            lambda: Rational(0, 1)
        )
        r = defaultdict(
            lambda: defaultdict(lambda: Rational(0, 1))
        )  # 初始化关系字典r，使用两层defaultdict确保默认值为Rational(0, 1)
        for i in range(n_cells):  # 遍历所有单元格
            cell_weight = cell_graph.get_cell_weight(cells[i])  # 获取当前单元格的权重
            t = list()  # # 初始化状态列表t，用于存储谓词状态 (1=true, 0=false)
            # 存在谓词
            for (
                pred
            ) in self.ext_preds:
                if cells[i].is_positive(pred):
                    t.append(0)
                else:
                    t.append(1)  # 添加状态0
            # 计数谓词
            for idx, (pred, param) in enumerate(zip(self.cnt_preds, self.cnt_params)):
                if cells[i].is_positive(pred):
                    if (
                        self.exist_mod and idx in self.mod_pred_index
                    ):
                        t.append(
                            self.cnt_remainder[idx] - 1
                        )
                    else:
                        t.append(param - 1)
                else:
                    if (
                        self.exist_mod and idx in self.mod_pred_index
                    ):
                        t.append(self.cnt_remainder[idx])
                    else:
                        t.append(param)
            w2t[i] = tuple(t)
            w[i] = w[i] + cell_weight

            for j in range(n_cells):
                cell1 = cells[i]
                cell2 = cells[j]
                for evi_idx, evidence in enumerate(
                    self.binary_evidence
                ):
                    t = list()
                    reverse_t = (
                        list()
                    )
                    two_table_weight = cell_graph.get_two_table_weight(
                        (cell1, cell2), evidence
                    )
                    if two_table_weight == Rational(
                        0, 1
                    ):
                        continue
                    for pred_idx, pred in enumerate(
                        self.ext_preds + self.cnt_preds
                    ):
                        if (
                            evi_idx >> (2 * pred_idx)
                        ) & 1 == 1:
                            reverse_t.append(1)
                        else:
                            reverse_t.append(0)
                        if (
                            evi_idx >> (2 * pred_idx + 1)
                        ) & 1 == 1:
                            t.append(1)
                        else:
                            t.append(0)
                    r[(i, j)][
                        (tuple(t), tuple(reverse_t))
                    ] = two_table_weight  # 使用谓词状态组合存储关系权重
        #         # --- START: 添加这段调试代码 ---
        # import logging
        # logging.basicConfig(level=logging.INFO) # 确保INFO级别的日志能被打印
        # logging.info("--- DEBUG: Two-table 'r' dictionary ---")
        # # 对 r 字典的键（cell对）进行排序，以保证打印顺序一致
        # for cell_pair, transitions in sorted(r.items()):
        #     # 只打印有内容的条目
        #     if transitions:
        #         logging.info(f"Cell Pair {cell_pair}:")
        #         # 对内部字典的键（状态转移）也排序
        #         for transition, weight in sorted(transitions.items()):
        #             if weight != 0: # 只打印非零权重的转移
        #                 logging.info(f"  Transition: {transition} -> Weight: {weight}")
        # logging.info("--- END DEBUG ---")
        # # --- END: 调试代码结束 ---
        return w2t, w, r  # 返回映射字典、权重字典和关系字典

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return default, default

    def check_unary_constraints(self, config, mask) -> tuple[bool, bool, bool]:
        return (
            self.check_unary_mod_constraints(config, mask[0]),
            self.check_unary_eq_constraints(config, mask[1]),
            self.check_unary_le_constraints(config, mask[2]),
        )

    def check_unary_mod_constraints(self, config, unary_mod_mask) -> bool:
        for mask, r_mod, k_mod in unary_mod_mask:
            config_total_unary_constraint = mask @ np.fromiter(
                config, dtype=np.int32
            )
            if config_total_unary_constraint % k_mod != r_mod:
                return True
        return False

    def check_unary_eq_constraints(self, config, unary_eq_mask) -> bool:
        for mask, k_eq in unary_eq_mask:
            if (mask @ np.fromiter(config, dtype=np.int32)) != k_eq:
                return True
        return False

    def check_unary_le_constraints(self, config, unary_le_masks) -> bool:
        vec = np.fromiter(config, dtype=np.int32)
        for mask, k_max in unary_le_masks:
            if (mask @ vec) > k_max:
                return True
        return False

    def build_unary_mask(self, cells) -> tuple[list, list, list]:
        return (
            self.build_unary_mod_mask(cells),
            self.build_unary_eq_mask(cells),
            self.build_unary_le_mask(cells),
        )

    def build_unary_le_mask(self, cells) -> list:
        n_cells = len(cells)
        masks = []
        for pred, k_max in self.unary_le_constraints:
            mask = np.fromiter(
                (1 if cell.is_positive(pred) else 0 for cell in cells),
                dtype=np.int8,
                count=n_cells,
            )
            masks.append((mask, k_max))
        return masks

    def build_unary_mod_mask(self, cells) -> list:

        n_cells = len(cells)
        masks = []
        for pred, r, k in self.unary_mod_constraints:
            mask = np.fromiter(
                (
                    1 if cell.is_positive(pred) else 0 for cell in cells
                ),
                dtype=np.int8,
                count=n_cells,
            )
            masks.append((mask, r, k))
        return masks

    def build_unary_eq_mask(self, cells) -> list:

        n_cells = len(cells)
        masks = []
        for pred, k_eq in self.unary_eq_constraints:
            mask = np.fromiter(
                (1 if cell.is_positive(pred) else 0 for cell in cells),
                dtype=np.int8,
                count=n_cells,
            )
            masks.append((mask, k_eq))
        return masks


class ConfigUpdater:
    def __init__(self, t_updates, shape, cache):
        self.t_updates = t_updates
        self.shape = shape
        self.Cache_F = cache

    def update_config(self, target_c, other_c, l):
        if (
            target_c,
            other_c,
        ) in self.Cache_F:
            config_updates_cache_num = self.Cache_F[
                (target_c, other_c)
            ]
            num_start = l
            while (
                num_start not in config_updates_cache_num and num_start > 0
            ):
                num_start -= 1
        else:
            self.Cache_F[(target_c, other_c)] = dict()
            num_start = 0
        if num_start == 0:
            F = dict()
            u_config = np.zeros(
                self.shape, dtype=np.uint8
            )
            u_config = HashableArrayWrapper(u_config)
            F[(target_c, u_config)] = Rational(
                1, 1
            )
        else:
            F = self.Cache_F[(target_c, other_c)][
                num_start
            ]
        for j in range(num_start + 1, l + 1):
            F_new = defaultdict(
                lambda: Rational(0, 1)
            )
            for (target_c_old, u), W in F.items():
                for (target_c_new, other_c_new), rij in self.t_updates[
                    (target_c_old, other_c)
                ].items():
                    F_config_new = np.array(u.array)
                    F_config_new[other_c_new] += 1
                    F_config_new = HashableArrayWrapper(
                        F_config_new
                    )
                    F_new[(target_c_new, F_config_new)] += (
                        W * rij
                    )
            F = F_new
            self.Cache_F[(target_c, other_c)][j] = F
        return F


class HashableArrayWrapper(object):
    def __init__(self, input_array: np.ndarray):
        self.array = input_array.astype(np.uint8, copy=False)

    def __hash__(self):
        return int(hashlib.sha1(self.array).hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return int(hashlib.sha1(self.array).hexdigest(), 16) == int(
                hashlib.sha1(other.array).hexdigest(), 16
            )
        return False

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return f"HashableArrayWrapper({self.array})"
