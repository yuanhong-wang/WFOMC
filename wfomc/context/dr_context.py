from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
from logzero import logger
from wfomc.fol.sc2 import SC2
from wfomc.fol.utils import new_predicate, convert_counting_formula
from wfomc.fol.syntax import AUXILIARY_PRED_NAME, X, Y, AtomicFormula, Const, Pred, QFFormula, top, a, b
from wfomc.network.constraint import CardinalityConstraint
from wfomc.fol.syntax import *
from wfomc.problems import WFOMCProblem
from wfomc.utils.third_typing import RingElement, Rational
from itertools import product


class DRWFOMCContext(object):
    """
    DRWFOMC算法的上下文类，用于存储和处理WFOMC问题的相关信息
    包括域、句子、权重、约束等信息，并提供预处理和构建辅助结构的方法
    """

    def __init__(self, problem: WFOMCProblem):
        """
        初始化DRWFOMC上下文

        Args:
            problem (WFOMCProblem): WFOMC问题实例
        """
        ## 域、句子、权重和基数约束
        self.domain: set[Const] = problem.domain
        self.sentence: SC2 = problem.sentence
        self.weights: dict[Pred, tuple[Rational, Rational]] = problem.weights
        self.cardinality_constraint: CardinalityConstraint = problem.cardinality_constraint
        self.repeat_factor = 1

        logger.info('sentence: \n%s', self.sentence)
        logger.info('domain: \n%s', self.domain)
        logger.info('weights:')
        for pred, w in self.weights.items():
            logger.info('%s: %s', pred, w)
        logger.info('cardinality constraint: %s', self.cardinality_constraint)

        self.formula: QFFormula  # 无量词公式
        ## 处理线性序公理
        if problem.contain_linear_order_axiom():
            self.leq_pred: Pred = Pred('LEQ', 2)
        else:
            self.leq_pred: Pred = None

        self.uni_formula = []  # 全称量词公式
        self.ext_preds = []  # 存在量词谓词列表

        # 单双层
        # 每层是mod = <=
        # 单层，谓词是一元
        # 双层 谓词是二元
        ## 计数
        self.cnt_preds = []  # 计数谓词列表
        self.cnt_params = []  # 计数参数 k (int)
        self.cnt_remainder = []  # 余数 r (int)
        ## unary
        self.mod_pred_index = []  # 模运算谓词索引
        self.exist_mod = False  # 是否存在模运算
        self.unary_mod_constraints = []  # 一元模约束 [(Pred, r, k), …]
        self.unary_eq_constraints = []  # [(pred, k), ...]
        self.unary_le_constraints = []  # [(pred, k_max), ...]

        ## <=
        self.exist_le = False  # "是否有<="
        self.le_pred = []  # 小于等于谓词列表
        self.le_index = []  # 小于等于谓词索引
        ## 比较器处理函数映射
        self.comparator_handlers = {  # 比较器处理函数映射
            'mod': self._handle_mod,
            '=': self._handle_eq,
            '<=': self._handle_le,  #
        }

        self._preprocess()  # 预处理逻辑公式

        self.c_type_shape = tuple()  # c类型形状
        self.build_c_type_shape()  # 构建c类型形状

        self.binary_evidence = []  # 二元证据
        self.get_binary_evidence()  # 获取二元证据

        # binary cardinality_constraints is underconstruction
        self.card_preds = []
        self.card_ccs = []
        self.card_vars = []
        self.build_cardinality_constraints()


    def stop_condition(self, last_target_c):
        """
        判断是否满足停止条件

        Args:
            last_target_c: 最后目标状态

        Returns:
            bool: 是否满足停止条件
        """
        if self.exist_le:
            # 获取位置索引index，然后index位置的元素值可以大于0，其余为0
            pred_state = last_target_c[1:]

            # 检查le_index索引位置的值是否>=0
            for idx in self.le_index:
                if idx < len(pred_state) and pred_state[idx] > 0:
                    return True

            # 检查非le_index位置是否都为0
            for i in range(len(pred_state)):
                if i not in self.le_index and pred_state[i] != 0:
                    return False
            return True
        else:
            return all(i == 0 for i in last_target_c[1:])

    def _split_layer(self, formula):
        """
        把计数公式拆成 (layer, inner_formula, qscope) 三元组
        layer ∈ {"Universal_Counting", "Counting"}
        inner_formula 一定是 P(X) 那一层

        Args:
            formula: 要拆分的公式

        Returns:
            tuple: (layer类型, 内部公式, 量词作用域)
        """
        if isinstance(formula.quantified_formula, QuantifiedFormula):
            # 双层 ∀X (∃_{·} Y : ...)
            inner_formula = formula.quantified_formula
            return ("double layer",
                    inner_formula.quantified_formula,  # P(X)
                    inner_formula.quantifier_scope)  # scope of ∃_{·} Y
        else:
            # 单层 ∃_{·} X : ...
            return ("single layer",
                    formula.quantified_formula,  # P(X)
                    formula.quantifier_scope)  # scope of ∃_{·} X

    def _add_aux_equiv(self, inner_formula):
        """
        创建二元辅助谓词并加进 self.uni_formula，同时返回 aux_pred
        Args:
            inner_formula: 内部公式
        Returns:
            Pred: 新创建的辅助谓词
        """
        aux_pred = new_predicate(2, AUXILIARY_PRED_NAME)  # 创建新的二元辅助谓词
        self.uni_formula = self.uni_formula & inner_formula.equivalent(aux_pred(X, Y))  # 添加等价关系: inner_formula ↔ aux(X,Y)
        self.cnt_preds.append(aux_pred)  # 将辅助谓词添加到计数谓词列表
        return aux_pred

    def _handle_mod(self, idx, inner_formula, qscope, param, _):  # 处理  ∃_{r mod k}
        """
        处理模运算量词 ∃_{r mod k}
        Args:
            idx: 索引
            inner_formula: 内部公式
            qscope: 量词作用域
            param: 参数 (r, k)
            _: 比较器（此处未使用）
        """
        r, k = param  # (r, k) 分解参数

        ## unary mod
        if (isinstance(inner_formula, AtomicFormula)
                and inner_formula.pred.arity == 1
                and inner_formula.args == (qscope.quantified_var,)):
            self.unary_mod_constraints.append((inner_formula.pred, r, k))  # 供 config 剪枝
            return  # 不再进入递归

        ## binary mod
        self.exist_mod = True
        self.mod_pred_index.append(idx) # 记录模运算谓词索引
        self.cnt_remainder.append(r) # 记录余数
        self.cnt_params.append(k) # 记录模数
        self._add_aux_equiv(inner_formula)  # 仍然引入二元 aux

    def _handle_eq(self, idx, inner_formula, qscope, param, comparator):
        """
        处理等号量词 ∃_{=m}
        Args:
            idx: 索引
            layer: 层级
            inner_formula: 内部公式
            qscope: 量词作用域
            param: 参数 k
            comparator: 比较器
        """
        ## unary ∃_{=k} X  A(X)
        if (isinstance(inner_formula, AtomicFormula)
                and inner_formula.pred.arity == 1
                and inner_formula.args == (qscope.quantified_var,)):
            self.unary_eq_constraints.append((inner_formula.pred, param))  # 供 config 剪枝
            return  # 不进递归，不占 cnt_preds
        ## biarny
        self.cnt_remainder.append(None)  # 不需要余数
        self.cnt_params.append(param)  # 直接添加参数k
        aux_pred = self._add_aux_equiv(inner_formula)  # 添加辅助等价关系

    def _handle_le(self, idx, inner_formula, qscope, param, comparator):
        """
        处理小于等于量词 ∃_{<=m}
        Args:
            idx: 索引
            layer: 层级
            inner_formula: 内部公式
            qscope: 量词作用域
            param: 参数 k
            comparator: 比较器
        """
        if (isinstance(inner_formula, AtomicFormula)
                and inner_formula.pred.arity == 1
                and inner_formula.args == (qscope.quantified_var,)):
            self.unary_le_constraints.append((inner_formula.pred, param))  # 供 config 剪枝
            return  # 不进递归，不占 cnt_preds

        self.cnt_remainder.append(None)  # 不需要余数
        self.cnt_params.append(param)  # 直接添加参数k
        aux_pred = self._add_aux_equiv(inner_formula)
        self.le_pred.append(aux_pred)  # 只有 <= 才需要额外标记
        self.exist_le = True  # 标记存在 <= 量词

    def _preprocess(self):
        """
        预处理逻辑公式，将其转换为可处理的无量词形式，并引入辅助谓词
        """
        ## 提取全称公式
        self.uni_formula = self.sentence.uni_formula
        while not isinstance(self.uni_formula, QFFormula):
            self.uni_formula = self.uni_formula.quantified_formula
        ext_formulas = self.sentence.ext_formulas  # 存在量词公式
        cnt_formulas = self.sentence.cnt_formulas  # 计数量词公式

        ## 处理存在公式
        for formula in ext_formulas:
            # NOTE: assume all existential formulas are of the form VxEy
            qf_formula = formula.quantified_formula.quantified_formula
            aux_pred = new_predicate(2, AUXILIARY_PRED_NAME)
            self.uni_formula = self.uni_formula & qf_formula.equivalent(aux_pred(X, Y))
            self.ext_preds.append(aux_pred)

        ## 处理计数公式
        for idx, formula in enumerate(cnt_formulas):
            _, inner_formula, qscope = self._split_layer(formula)  # 因为可能是双层或单层，所以需要拆分
            comparator = qscope.comparator  # 'mod' / '=' / '<=' / ...
            cnt_param_raw = qscope.count_param  # (r,k) 或 int

            ## 根据 comparator 分派到对应的 handler
            idx = len(self.cnt_preds)  # 用当前 cnt_preds 长度算下标
            # 也就是说，cnt_formulas 和 cnt_preds的长度是不同的。unary mod不会添加进cnt_preds中。为了跳过unary mod,不采用手动累加 idx = idx + 1。是因为idx 必须始终与 cnt_preds 的当前长度保持同步，保持新谓词下标依然连续、正确，
            self.comparator_handlers[comparator](idx, inner_formula, qscope, cnt_param_raw, comparator)

        self.all_preds = self.ext_preds + self.cnt_preds # 收集全部谓词
        self._pred2idx = {pred: i for i, pred in enumerate(self.all_preds)} # 构建一次性的“谓词 → 索引”映射（O(n)）
        self.le_index = [self.all_preds.index(pred) for pred in self.le_pred] # 利用映射拿 <= 量词对应的索引

    def build_c_type_shape(self):
        """
        构建c类型形状，用于表示状态空间的维度
        """
        self.c_type_shape = list(2 for _ in self.ext_preds)  # 存在谓词的状态空间为2（真假）
        for idx, k in enumerate(self.cnt_params):  # 为每个计数参数构建状态空间
            if idx in self.mod_pred_index:  # 如果是模k类型的量词 ∃_{r mod k}
                self.c_type_shape.append(k)  # 状态空间为 0 到 k-1
            else:  # 是∃_{k}类型
                self.c_type_shape.append(k + 1)  # 状态空间为 0 到 k

    def get_binary_evidence(self):
        """
        获取二元证据，用于构建权重和关系字典
        """
        ext_atoms = list(
            ((~pred(a, b), ~pred(b, a)),
             (~pred(a, b), pred(b, a)),
             (pred(a, b), ~pred(b, a)),
             (pred(a, b), pred(b, a)))
            for pred in self.ext_preds[::-1])  # 反向遍历存在谓词
        cnt_atoms = list(
            ((~pred(a, b), ~pred(b, a)),
             (~pred(a, b), pred(b, a)),
             (pred(a, b), ~pred(b, a)),
             (pred(a, b), pred(b, a)))
            for pred in self.cnt_preds[::-1])  # 反向遍历计数谓词
        # 通过笛卡尔积生成所有可能的原子公式组合
        for atoms in product(*cnt_atoms, *ext_atoms):
            self.binary_evidence.append(frozenset(sum(atoms, start=())))  # 添加到二元证据列表

    def build_cardinality_constraints(self):  # this code is under construction
        if self.contain_cardinality_constraint():
            pred2var = dict((pred, var) for var, pred in self.cardinality_constraint.var2pred.items())
            constraints = self.cardinality_constraint.constraints
            for constraint in constraints:
                coeffs, comp, param = constraint
                assert len(coeffs) == 1 and comp == '='
                param = int(param)
                pred, coef = next(iter(coeffs.items()))
                self.card_preds.append(pred)
                assert coef == 1
                self.card_ccs.append(param)
                # self.card_vars.append(pred2var[pred])

    def contain_cardinality_constraint(self) -> bool:
        """
        判断是否包含基数约束

        Returns:
            bool: 是否包含基数约束
        """
        return self.cardinality_constraint is not None and \
            not self.cardinality_constraint.empty()

    def build_t_updates(self, r, n_cells, domain_size):
        """
        构建状态转移权重表

        用于计算在给定关系字典r的情况下，不同单元格状态之间的转移权重

        Args:
            r: 关系字典，包含单元格对之间的关系和权重
            n_cells: 单元格数量
            domain_size: 论域大小

        Returns:
            defaultdict: 状态转移权重表 t_updates，外层键是当前联合状态(c1, c2)，
                         内层键是下一个联合状态(c1_new, c2_new)，值是转移权重
        """
        t_updates = defaultdict(lambda: defaultdict(
            lambda: Rational(0,
                             1)))  # 初始化一个两层嵌套的defaultdict： 外层key是(c1, c2)表示当前联合状态 内层key是(c1_new, c2_new)表示下一个联合状态，value是转移权重，初始化为有理数0/1 # Initialize a two-level defaultdict: outer key is (c1, c2) (current joint state); inner key is (c1_new, c2_new) (next joint state); value is transition weight

        # construct c-type below

        if self.exist_mod:  # 如果存在模运算谓词
            final_list = [tuple(range(2)) for _ in self.ext_preds]  # 为存在谓词构建状态空间（每个存在谓词有两个状态：0和1）
            for idx, k in enumerate(self.cnt_params):  # 为计数谓词构建状态空间
                if idx in self.mod_pred_index:  # 如果当前计数谓词是模运算谓词
                    # final_k = (domain_size // k) * k  # 计算一个不大于domain_size的最大k的倍数， mod 2 -> 2,4,6...
                    # final_list += [tuple(range(final_k + 1))]  # 添加状态空间（0到final_k）
                    final_list += [tuple(range(k))]
                else:  # 非模运算谓词，状态空间为0到k
                    final_list += [tuple(range(k + 1))]
                # print("final_list-->>",final_list)
            all_ts = list(product(*(final_list)))  # 通过笛卡尔积生成所有可能的状态组合
            # print("all_ts-->>",all_ts)
        else:
            # 枚举单个单元格的所有有效内部状态：
            # 1. tuple(range(2)) 为每个存在谓词提供值集 {0,1}；
            # 2. tuple(range(k+1)) 为每个计数谓词提供 {0,…,k}；
            # 3. 连接这些列表并做笛卡尔积 (itertools.product) 得到所有组合 all_ts 类似 [(b1,…,bn, c1,…,cm), …]。
            all_ts = list(
                product(*([tuple(range(2)) for _ in self.ext_preds] + [tuple(range(i + 1)) for i in self.cnt_params]))
            )
        # 双重循环遍历有序单元对 (i,j)（允许 i=j，相同单元对）
        for i in range(n_cells):
            for j in range(n_cells):
                for t1 in all_ts:  # 枚举源单元格 i 的当前状态 t1 和目标单元格 j 的当前状态 t2
                    for t2 in all_ts:
                        # 遍历关系字典 r[(i,j)] 中注册的所有状态增量
                        # • dt 应用于 t1
                        # • reverse_dt 应用于 t2
                        # • rijt 是这个增量的权重（概率/贡献）
                        for (dt, reverse_dt), rijt in r[(i, j)].items():
                            # 1) 首先做"旧式减法"一次（应用状态增量）
                            t1_new = [a - b for a, b in zip(t1, dt)]
                            t2_new = [a - b for a, b in zip(t2, reverse_dt)]

                            # === 2) 对于计数谓词，区分两种语义 ===
                            if self.exist_mod:  # 如果存在模运算谓词
                                for p, k_i in enumerate(self.cnt_params):  # 遍历所有计数参数 # p = 0,1,…  (局部下标)
                                    index = len(self.ext_preds) + p  # 在 t1_new / t2_new 里的位置，因为状态元组的结构是：(存在谓词状态, 计数谓词状态)，所以需要跳过所有存在谓词的位置。
                                    if p in self.mod_pred_index:  # ← 用局部下标做判断
                                        # mod-k：把 -1 折回 (k-1)，其它值取模。
                                        # 举个例子，假设k=3（模3运算）：
                                        # 如果当前值是0，减1后变成-1，根据模运算规则，-1 ≡ 2 (mod 3)，所以变成2
                                        # 如果当前值是5，减1后变成4，4 % 3 = 1，所以变成1
                                        if t1_new[index] == -1:
                                            t1_new[index] = k_i - 1
                                        else:
                                            t1_new[index] %= k_i
                                        if t2_new[index] == -1:
                                            t2_new[index] = k_i - 1
                                        else:
                                            t2_new[index] %= k_i
                                    else:
                                        # =k：出现负数就是非法，直接跳过这一条转移
                                        if t1_new[index] < 0 or t2_new[index] < 0:
                                            continue
                            else:  # 只有普通计数量化器 ∃=k (无 mod-k): 负数是无效的
                                if any(
                                        t1_new[len(self.ext_preds) + p] < 0 or
                                        t2_new[len(self.ext_preds) + p] < 0
                                        for p in range(len(self.cnt_params))
                                ):
                                    continue

                            # 修正布尔状态（确保非负）
                            for idx in range(
                                    len(self.ext_preds)):  # 存在量词谓词状态（布尔值）必须非负：例如，t1_new = [1, -1, 1] → 修正为 [1, 0, 1]
                                t1_new[idx] = max(t1_new[idx], 0)
                                t2_new[idx] = max(t2_new[idx], 0)

                            # 写入 t_updates
                            c1 = (i,) + t1  # 当前源单元格状态 (i, t1)
                            c2 = (j,) + t2  # 当前目标单元格状态 (j, t2)
                            c1_new = (i,) + tuple(t1_new)  # 新的源单元格状态
                            c2_new = (j,) + tuple(t2_new)  # 新的目标单元格状态
                            t_updates[(c1, c2)][(c1_new, c2_new)] += rijt  # 累加转移权重到状态转移表中
        return t_updates

    def build_weight(self, cells, cell_graph):
        """
        构建权重和关系字典 r

        该函数根据给定的单元格和单元格图计算权重字典和关系字典，
        用于后续的状态转移计算

        Args:
            cells: 单元格类型列表
            self.ext_preds: 存在量词谓词列表
            self.cnt_preds: 计数谓词列表
            self.cnt_params: 计数参数列表
            self.binary_evidence: 二元证据列表
            cell_graph: 单元格图对象

        Returns:
            tuple: (w2t 字典, w 权重字典, r 关系字典)
                   w2t: 从单元格索引到谓词状态字典的映射, 例如 {0: (1, 0, 1, 2), 1: (0, 1, 0, 1)}, 值代表需要满足的数量
                   w: 每个单元格类型的权重字典，例如 {0: Rational(1, 1), 1: Rational(2, 1)}, 键是单元格索引，值是权重
                   r: 单元格对之间的关系字典，
        """

        n_cells = len(cells)
        w2t = dict()  # 初始化w2t字典，用于映射单元格索引到谓词状态
        w = defaultdict(lambda: Rational(0, 1))  # 初始化权重字典w，使用defaultdict确保默认值为Rational(0, 1)
        r = defaultdict(lambda: defaultdict(lambda: Rational(0, 1)))  # 初始化关系字典r，使用两层defaultdict确保默认值为Rational(0, 1)
        for i in range(n_cells):  # 遍历所有单元格
            cell_weight = cell_graph.get_cell_weight(cells[i])  # 获取当前单元格的权重
            t = list()  # # 初始化状态列表t，用于存储谓词状态 (1=true, 0=false)
            ## 存在谓词
            for pred in self.ext_preds:  # 对存在谓词 (self.ext_preds)，存储 1 表示正，0 表示负
                if cells[i].is_positive(pred):  # 如果当前单元格中该谓词为正
                    t.append(1)  # 添加状态1
                else:
                    t.append(0)  # 添加状态0
            ## 计数谓词
            for idx, (pred, param) in enumerate(zip(self.cnt_preds, self.cnt_params)):
                if cells[i].is_positive(pred):  # 如果当前单元格中该计数谓词为正
                    if self.exist_mod and idx in self.mod_pred_index:  # 检查是否存在模运算且当前谓词是模运算谓词
                        t.append(self.cnt_remainder[idx] - 1)  # 如果是模运算谓词，存储余数减1
                    else:  # 普通计数谓词，存储参数值减1
                        t.append(param - 1)
                else:  # 如果当前单元格中该计数谓词为负
                    if self.exist_mod and idx in self.mod_pred_index:  # 检查是否存在模运算且当前谓词是模运算谓词
                        t.append(self.cnt_remainder[idx])  # 如果是模运算谓词，存储余数
                    else:  # 普通计数谓词，存储参数值
                        t.append(param)
            w2t[i] = tuple(t)  # 将当前单元格的状态元组存储到w2t映射中
            w[i] = w[i] + cell_weight  # 累积单元格权重到权重字典中

            for j in range(n_cells):  # 遍历所有单元格对
                cell1 = cells[i]  # 获取第一个单元
                cell2 = cells[j]  # 获取第二个单元格
                for evi_idx, evidence in enumerate(self.binary_evidence):  # 遍历所有可能的二元证据
                    t = list()  # t: 存储当前证据中每个谓词的"正向"状态 (a→b 方向)
                    reverse_t = list()  # reverse_t: 存储当前证据中每个谓词的"反向"状态 (b→a 方向)
                    two_table_weight = cell_graph.get_two_table_weight(  # 获取两个单元格之间的二元表权重
                        (cell1, cell2), evidence  # 传入单元格对和证据
                    )
                    if two_table_weight == Rational(0, 1):  # 如果权重为零（无效配置），则跳过
                        continue
                    for pred_idx, pred in enumerate(self.ext_preds + self.cnt_preds):  # 检查两个方向的谓词状态
                        if (evi_idx >> (2 * pred_idx)) & 1 == 1:  # 根据证据索引的位模式确定反向状态
                            reverse_t.append(1)
                        else:
                            reverse_t.append(0)
                        if (evi_idx >> (2 * pred_idx + 1)) & 1 == 1:  # 根据证据索引的位模式确定正向状态
                            t.append(1)
                        else:
                            t.append(0)
                    r[(i, j)][(tuple(t), tuple(reverse_t))] = two_table_weight  # 使用谓词状态组合存储关系权重
        return w2t, w, r  # 返回映射字典、权重字典和关系字典

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return default, default

    def check_unary_constraints(self, config, mask) -> tuple[bool,bool, bool]:
        return self.check_unary_mod_constraints(config, mask[0]), self.check_unary_eq_constraints(config, mask[1]), self.check_unary_le_constraints(config, mask[2])


    def check_unary_mod_constraints(self, config, unary_mod_mask)-> bool:
        for mask, r_mod, k_mod in unary_mod_mask:  # 遍历每个约束
            config_total_unary_constraint = (mask @ np.fromiter(config, dtype=np.int32))  # config 是当前 1-type 配置，元素是“第 i 个 cell 放了多少个元素”。mask @ config 就是向量点积 —— 自动算出 整个结构里满足 pred 的元素总数
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
            if (mask @ vec) > k_max: # count > k ⇒ 违反
                return True
        return False

    def build_unary_mask(self, cells) -> tuple[list, list, list]:
        return self.build_unary_mod_mask(cells),self.build_unary_eq_mask(cells), self.build_unary_le_mask(cells)
    def build_unary_le_mask(self, cells) -> list:
        n_cells = len(cells)
        masks = []
        for pred, k_max in self.unary_le_constraints:
            mask = np.fromiter(
                (1 if cell.is_positive(pred) else 0 for cell in cells),
                dtype=np.int8, count=n_cells
            )
            masks.append((mask, k_max))
        return masks

    def build_unary_mod_mask(self, cells) -> list:
        """
        # 一阶 1-type cell 是否把某个一元谓词 pred 标成 True”转换成一个长度为 n_cells 的 0‒1 向量mask。比如，cells = [B(X)^LEQ(X,X)^~@aux0(X,X)^~A(X), @aux0(X,X)^A(X)^LEQ(X,X)^~B(X)], 那么unary_mod_mask = [([0 1], 0, 2)]
        """
        n_cells = len(cells)
        masks = [] # 每个约束对应一个 mask 和 (r,k) [(np.int8[n_cells], r, k), …]
        for pred, r, k in self.unary_mod_constraints:
            mask = np.fromiter(
                (1 if cell.is_positive(pred) else 0 for cell in cells), # cell 是否满足该一元谓词
                dtype=np.int8,
                count=n_cells
            )
            masks.append((mask, r, k))
        return masks

    def build_unary_eq_mask(self, cells) -> list:
        """
        Parameters
        ----------
        cells : List[OneTypeCell]
            全部 1-type cell，对应 domain_recursive_wfomc 里的 `cells`

        Returns
        -------
        List[(np.ndarray[int8], int)]
            (mask, k_eq)；mask 长度 = n_cells，值 ∈ {0,1}
        """
        n_cells = len(cells)
        masks = []
        for pred, k_eq in self.unary_eq_constraints:
            mask = np.fromiter(
                (1 if cell.is_positive(pred) else 0 for cell in cells),
                dtype=np.int8,
                count=n_cells
            )
            masks.append((mask, k_eq))
        return masks


class ConfigUpdater:
    def __init__(self, t_updates, shape, cache):
        self.t_updates = t_updates
        self.shape = shape
        self.Cache_F = cache  # global Cache_F

    def update_config(self, target_c, other_c, l):
        if (target_c, other_c) in self.Cache_F:  # 检查缓存中是否已存在目标单元格和另一个单元格的配置更新
            config_updates_cache_num = self.Cache_F[(target_c, other_c)]  # 获取该单元格对的缓存配置更新数据
            num_start = l  # 从l开始向下查找最大的已缓存索引j ≤ l
            while num_start not in config_updates_cache_num and num_start > 0:  # 继续查找直到找到已缓存的索引或到达0
                num_start -= 1
        else:  # 如果该单元格对没有缓存，则初始化缓存
            self.Cache_F[(target_c, other_c)] = dict()
            num_start = 0  # 设置起始索引为0
        # 初始化F（状态转移权重字典）
        if num_start == 0:
            F = dict()  # 创建新的状态字典
            u_config = np.zeros(self.shape, dtype=np.uint8)  # 创建一个形状为self.shape的零数组作为初始配置
            u_config = HashableArrayWrapper(u_config)  # 将numpy数组包装为可哈希数组
            F[(target_c, u_config)] = Rational(1, 1)  # 初始权重为1，表示目标单元格在无配置更改时的初始状态
        else:
            F = self.Cache_F[(target_c, other_c)][num_start]  # 如果存在缓存，则从缓存中获取起始状态
        # 主循环：从num_start+1迭代到l
        for j in range(num_start + 1, l + 1):
            F_new = defaultdict(lambda: Rational(0, 1))  # 为当前迭代创建新的状态字典，使用默认值为有理数0/1的defaultdict
            for (target_c_old, u), W in F.items():  # 处理每个现有的状态转移
                for (target_c_new, other_c_new), rij in self.t_updates[(target_c_old, other_c)].items():  # 遍历所有可能的状态转移，获取新状态和转移权重
                    F_config_new = np.array(u.array)  # 复制当前配置数组
                    F_config_new[other_c_new] += 1  # 增加对应other_c_new状态的计数
                    F_config_new = HashableArrayWrapper(F_config_new)  # 将更新后的数组重新包装为可哈希数组
                    F_new[(target_c_new, F_config_new)] += W * rij  # 累加转移权重到新状态
            F = F_new  # 更新F为新状态字典
            self.Cache_F[(target_c, other_c)][j] = F  # 缓存当前迭代的结果
        return F  # 返回最终的状态转移权重字典




class HashableArrayWrapper(object):
    def __init__(self, input_array: np.ndarray):
        self.array = input_array.astype(np.uint8, copy=False)

    def __hash__(self):
        return int(hashlib.sha1(self.array).hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return int(hashlib.sha1(self.array).hexdigest(), 16) == \
                int(hashlib.sha1(other.array).hexdigest(), 16)
        return False

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return f"HashableArrayWrapper({self.array})"


