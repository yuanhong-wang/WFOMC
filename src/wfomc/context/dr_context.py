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


class DRWFOMCContext(object):
    def __init__(self, problem: WFOMCProblem): # problem (WFOMCProblem): WFOMC问题实例
        # 域、句子、权重和基数约束
        self.problem: WFOMCProblem = deepcopy(
            problem)  # 深拷贝传入的WFOMC问题，避免修改原始对象。
        self.domain: set[Const] = problem.domain  # 域
        self.sentence: SC2 = problem.sentence  # 逻辑句子: 提取逻辑句子，格式为SC2。
        self.weights: dict[Pred, tuple[Rational, Rational]
                           ] = problem.weights  # 权重
        self.cardinality_constraint: CardinalityConstraint = (
            problem.cardinality_constraint
        )  # 基数约束
        self.repeat_factor: int = 1  # 重复因子，默认值为1: 初始化一个重复因子，用于某些特殊情况下的计数调整。

        logger.info("sentence: \n%s", self.sentence)
        logger.info("domain: \n%s", self.domain)
        logger.info("weights:")
        for pred, w in self.weights.items():
            logger.info("%s: %s", pred, w)
        logger.info("cardinality constraint: %s", self.cardinality_constraint)

        self.formula: QFFormula  # 无量词公式: 声明一个实例变量，用于存储处理后的无量词公式。
        # --- 处理线性序公理
        if problem.contain_linear_order_axiom():  # 检查问题是否包含线性序公理。
            self.leq_pred: Pred = Pred("LEQ", 2)  # 如果是，则定义一个名为"LEQ"的二元谓词来表示它。
        else:
            self.leq_pred: Pred = None
        self.uni_formula: QuantifiedFormula = Top # 全称公式: 初始化一个变量来存储公式的纯全称部分，初始为"Top" (真)。
        self.ext_preds: list[QuantifiedFormula] = []  # 存在量词谓词列表
        # --- cc 处理计数量词，分为单双层，每层是mod = <=。单层即谓词是一元，双层即谓词是二元
        self.cnt_preds: list[QuantifiedFormula] = [] # 计数谓词列表: 初始化列表，存储为处理计数量词而引入的辅助谓词。
        self.cnt_params: list[int] = [] # 计数参数 k (int): 初始化列表，存储计数量词的参数（例如，∃=k 中的 k）。
        self.cnt_remainder: list[int] = [] # 余数 r (int): 初始化列表，存储模数计数量词的余数（例如，∃≡r (mod k) 中的 r）。
        # --- unary 一元约束相关的变量
        self.mod_pred_index: list[int] = []  # 模运算谓词索引
        self.exist_mod: bool = False  # 是否存在模运算
        self.unary_mod_constraints: list[tuple] = []  # 一元模约束 [(Pred, r, k), …]
        self.unary_eq_constraints: list[tuple] = []  # [(pred, k), ...]
        self.unary_le_constraints: list[tuple] = []  # [(pred, k_max), ...]
        # --- 处理<=
        self.exist_le: bool = False  # "是否有<="
        self.le_pred: list[Pred] = []  # 小于等于谓词列表
        self.le_index: list[int] = []  # 小于等于谓词索引
        # 比较器处理函数映射
        self.comparator_handlers: dict[str, callable] = {  # 比较器处理函数映射
            "mod": self._handle_mod,
            "=": self._handle_eq,
            "<=": self._handle_le,
        }
        self._build()  # 预处理逻辑公式: 调用_build方法，开始对逻辑公式进行转换和分解。
        self.c_type_shape: tuple = tuple()
        self.build_c_type_shape()
        self.binary_evidence: list = []
        self.build_binary_evidence()
        # binary cardinality_constraints is underconstruction 这部分采用的是symbolic weight
        self.card_preds: list = []
        self.card_ccs: list = []
        self.card_vars: list = []
        self.build_cardinality_constraints()
        self.build_repeat_factor() # 更新 repeat factor, 比如 m-odd 这个输入的例子 需要除以 domain size，也就是n choose 1。

    def build_repeat_factor(self):
        """
        由于odd-degree 这个输入的例子 需要除以 domain size
        查找odd 和 U 这两个一元等号约束是否在约束中出现，如果同时存在，则将 repeat_factor 设置为 domain size
        """
        if hasattr(self, 'unary_eq_constraints') and self.unary_eq_constraints: 
            constraint_names = {
                constraint[0].name for constraint in self.unary_eq_constraints}
            if 'Odd' in constraint_names and 'U' in constraint_names:
                self.repeat_factor = len(self.problem.domain)
                print("change repeat factor to:", self.repeat_factor)

    def stop_condition(self, target_c):
        """
        检查目标元素的最终状态是否满足所有约束。
        Args:
            target_c: target 元素的 c type

        Returns:
            bool: 
        """
        pred_state = target_c[1:] # 获取元素当前的c type，也就是还需要被连接的数量
        if self.exist_le: # 检查问题是否包含 <=k类型的计数量词，因为它们的处理逻辑不同。
            for i in range(len(pred_state)): 
                if i not in self.le_index and pred_state[i] != 0: # 如果当前索引 i 不属于宽松的 <=k约束, 并且计数状态不为零
                    return False # 则说明该严格约束未被满足，立即判定失败。
        else: # --- 不存在 <=k约束 ---
            return all(i == 0 for i in pred_state) # 查看target 对应的c type, 是否不 需要连接

    def _extract_formula(self, formula):
        """
        提取计数量词公式的类型、内核公式和量词作用域
        """
        # 这一行检查传入公式 formula内部的 quantified_formula属性是否仍然是一个 QuantifiedFormula 类的实例。如果是，说明这是一个嵌套的量词公式（即“二元”结构）。
        if isinstance(formula.quantified_formula, QuantifiedFormula):
            inner_formula = formula.quantified_formula  # 将内部的有量词公式赋值给 inner_formula 变量。
            return (
                "binary",  # "binary": 字符串，标记这是一个二元/嵌套类型的公式。
                inner_formula.quantified_formula,  # 提取不带量词的“内核”公式。
                inner_formula.quantifier_scope,  # 提取该量词的作用域信息。
            )
        else:
            return (
                "unary",  # "unary": 字符串，标记这是一个一元/单层类型的公式。
                formula.quantified_formula,  # 提取不带量词的“内核”公式。
                formula.quantifier_scope,  # 提取该量词的作用域信息。
            )

    def _add_aux_equiv(self, inner_formula):
        aux_pred = new_predicate(2, AUXILIARY_PRED_NAME) # 创建一个新的二元辅助谓词
        self.uni_formula = self.uni_formula & inner_formula.equivalent(
            aux_pred(X, Y)
        ) # 将等价关系添加到主公式中
        self.cnt_preds.append(aux_pred) # 添加辅助谓词到计数谓词列表
        return aux_pred

    def _handle_mod(self, type, idx, inner_formula, qscope, param, _):
        """
        处理模运算量词 ∃_{≡r (mod k)}
        Args:
            type: unary 还是 binary
            idx: 索引
            inner_formula: 内部公式
            qscope: 量词作用域
            param: 参数 (r, k)
            comparator: 比较器
        """
        r, k = param
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
            self.exist_mod = True # 存在模运算
            self.mod_pred_index.append(idx) # 记录模运算谓词索引
            self.cnt_remainder.append(r) # 记录余数 r
            self.cnt_params.append(k) # 记录参数 k
            self._add_aux_equiv(inner_formula) # 添加辅助等价谓词

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
        ): # 一元等号约束
            self.unary_eq_constraints.append(
                (inner_formula.pred, param)
            ) # 记录一元等号约束 
            return # 注意，这种谓词不需要添加进 cnt_preds 列表，因为它们在算法中是作为单独的约束处理的。
        elif type == "binary":
            self.cnt_remainder.append(None) # 二元等号没有余数
            self.cnt_params.append(param) # 记录参数 k
            self._add_aux_equiv(inner_formula) # 添加辅助等价谓词

    def _handle_le(self, type, idx, inner_formula, qscope, param, comparator):
        if (
            type == "unary"
            and isinstance(inner_formula, AtomicFormula)
            and inner_formula.pred.arity == 1
            and inner_formula.args == (qscope.quantified_var,)
        ): # under construction
            self.unary_le_constraints.append(
                (inner_formula.pred, param)
            )
            return
        elif type == "binary": # binary <=
            self.cnt_remainder.append(None) # 二元<=没有余数
            self.cnt_params.append(param) # 记录参数 k
            aux_pred = self._add_aux_equiv(inner_formula) # 添加辅助等价谓词
            self.le_pred.append(aux_pred) # 记录小于等于谓词
            self.exist_le = True # 标记存在 <= 约束

    def _build(self):
        """
        构建与预处理核心函数。

        此方法负责将原始的 SC2 逻辑句子分解并转换为算法可处理的内部表示。主要执行以下操作：
        1.  提取公式中的纯全称部分（无量词范式）。
        2.  对存在量词（ext_formulas）进行 Skolem 化，引入新的 Skolem 谓词，并将约束添加到主公式中。
        3.  对计数量词（cnt_formulas）进行分类处理：
            -   一元计数量词被提取为独立约束（如 unary_eq_constraints）。
            -   二元计数量词通过引入辅助谓词（auxiliary predicates）来处理，并将等价关系添加到主公式中。
        4.  整合所有新生成的辅助谓词（Skolem 谓词和计数谓词），并为它们建立索引。
        """
        # 1. 提取纯全称无量词范式 (QF-Form)
        # 循环剥离最外层的全称量词，直到剩下核心的无量词公式。
        self.uni_formula = self.sentence.uni_formula
        while not isinstance(self.uni_formula, QFFormula):
            self.uni_formula = self.uni_formula.quantified_formula

        # 获取句子中的存在量词和计数量词部分
        ext_formulas = self.sentence.ext_formulas  # 存在量词公式
        cnt_formulas = self.sentence.cnt_formulas  # 计数量词公式

        # 2. 处理存在量词 (Skolem化)
        # 遍历所有存在量词公式，通过Skolem化将其消除，并将生成的Skolem约束(&)合并到主公式中。
        for formula in ext_formulas:
            self.uni_formula = self.uni_formula & self._skolemize_one_formula(
                formula)

        # 3. 处理计数量词
        # 遍历所有计数量词公式，根据其类型（一元/二元）和比较符（=, <=, mod）进行分发处理。
        for idx, formula in enumerate(cnt_formulas):
            # 解析计数量词的结构：类型（一元/二元）、内核公式、量词作用域
            type, inner_formula, qscope = self._extract_formula(
                formula
            )  # 返回单双层类型type, 内核公式inner_formula, 量词作用域qscope
            comparator = qscope.comparator  # 获取比较符 'mod' / '=' / '<=' / ...
            cnt_param_raw = qscope.count_param  # 获取计数参数 (r,k) 或 int

            # 使用分派表 self.comparator_handlers 调用对应的处理函数 (e.g., _handle_eq)。
            # 注意：idx 使用 self.cnt_preds 的当前长度。这是因为一元约束不会增加新的计数谓词到cnt_preds 列表中，此方法可确保为新的二元计数谓词分配连续且正确的索引。
            idx = len(self.cnt_preds)  # 用当前 cnt_preds 长度作为下标
            # 注意，cnt_formulas 和 cnt_preds的长度是不同的。unary mod不会添加进cnt_preds中。为了跳过unary mod,不采用手动累加 idx = idx + 1。是因为idx 必须始终与 cnt_preds 的当前长度保持同步，保持新谓词下标依然连续、正确，
            self.comparator_handlers[comparator](
                type, idx, inner_formula, qscope, cnt_param_raw, comparator
            )

        # 4. 整合所有辅助谓词并建立索引
        # 合并Skolem化产生的新谓词 (ext_preds) 和处理二元计数量词产生的新谓词 (cnt_preds)。
        self.all_preds = self.ext_preds + self.cnt_preds
        self._pred2idx = {
            pred: i for i, pred in enumerate(self.all_preds)
        }  # 然后创建一个从谓词对象到其在总列表中索引的映射字典 _pred2idx，方便后续快速查找。
        self.le_index = [
            self.all_preds.index(pred) for pred in self.le_pred
        ]  # 如果公式中包含 <=k 类型的计数量词，它们对应的辅助谓词会被记录在 self.le_pred 中。这一行代码的作用是找出这些特殊谓词在 self.all_preds 总列表中的索引，并保存起来，因为算法对它们有特殊的处理逻辑。

    def _skolemize_one_formula(self, formula: QuantifiedFormula) -> QFFormula:
        """对单个存在量词公式进行Skolem化。

        Skolem化是一种消除存在量词的标准逻辑技术。此函数将一个形如 ∃Y: φ(X,Y) 的公式
        转换为一个逻辑上等价（在可满足性意义上）的、不含存在量词的公式。
        它通过引入一个新的Skolem谓词 S 来实现这一点，并生成约束，例如 φ(X,Y) → S(X)。

        主要步骤:
        1.  解析输入公式，确定存在量词的嵌套深度（quantifier_num）。
        2.  如果内核公式 φ 本身是复杂的，则引入一个辅助谓词 @aux 来简化它，并添加等价约束 (φ ↔ @aux)。
        3.  根据外部变量（如此处的 X）创建一个新的Skolem谓词 S。
        4.  添加核心的Skolem约束，形式为 S ∨ ¬φ (等价于 φ → S)。
        5.  为新的Skolem谓词 S 设置一个特殊的权重 (1, -1)，这是一种通过权重机制强制满足逻辑约束的技巧。

        Args:
            formula (QuantifiedFormula): 一个待处理的存在量词公式。

        Returns:
            QFFormula: Skolem化后生成的不含存在量词的约束公式。
        """
        quantified_formula = formula.quantified_formula  # 获取量词内部的公式，例如对于 ∃Y: P(X,Y)，获取的是 P(X,Y)。
        quantifier_num = 1  # 初始化量词数量为1。这个变量用来记录存在量词的层数，以确定Skolem谓词的元数（arity）。
        while not isinstance(
            quantified_formula, QFFormula
        ):  # 这个循环用于处理嵌套的存在量词，例如 ∃Y ∃Z: P(X,Y,Z)。
            quantified_formula = quantified_formula.quantified_formula
            quantifier_num += 1  # 它会持续剥离量词，直到找到最内层的无量词公式(QFFormula)，并计算剥离的层数。

        skolem_formula: QFFormula = top  # 初始化一个空的Skolem公式，初始值为 "Top" (逻辑真)。

        # ext_formula 指的是存在量词内核的无量词公式部分，例如 P(X,Y)
        ext_formula = quantified_formula
        if not isinstance(
            ext_formula, AtomicFormula
        ):  # 如果内核公式不是一个简单的原子公式（例如，它是一个复合公式 A(X) & B(Y)），则需要引入一个辅助谓词Z来简化它。
            aux_pred = new_predicate(
                quantifier_num, AUXILIARY_PRED_NAME
            )  # 创建一个新的辅助谓词，其元数由量词数量决定。
            aux_atom = (
                aux_pred(X, Y) if quantifier_num == 2 else aux_pred(X)
            )  # 根据元数创建原子公式，例如 aux(X,Y) 或 aux(X)。
            skolem_formula = skolem_formula & (
                ext_formula.equivalent(aux_atom)
            )  # 将 "内核公式 <=> 辅助原子" 这个等价约束添加到Skolem公式中。
            ext_formula = (
                aux_atom
            )  # 后续处理将直接使用这个更简单的辅助原子公式。

        # 根据量词数量创建Skolem谓词和对应的原子。
        # 这里的元数取决于外部全称量词的变量数量，代码中假设最多为1个（即X）。
        if quantifier_num == 2:  # 对应 ∀X ∃Y ...
            skolem_pred = new_predicate(
                1, SKOLEM_PRED_NAME
            )  # 创建一个一元Skolem谓词 S(X)。
            skolem_atom = skolem_pred(X)
        elif quantifier_num == 1:  # 对应 ∃Y ... (没有外部全称量词)
            skolem_pred = new_predicate(
                0, SKOLEM_PRED_NAME
            )  # 创建一个零元Skolem谓词 S()，即一个命题。
            skolem_atom = skolem_pred()

        skolem_formula = skolem_formula & (
            skolem_atom | ~ext_formula
        )  # 这等价于 P(X,Y) → S(X)。
        self.weights[skolem_pred] = (
            Rational(1, 1),
            Rational(-1, 1),
        )  # 为新创建的Skolem谓词设置权重。
        return skolem_formula

    def build_c_type_shape(self):
        """构建计数状态空间的维度信息"""
        self.c_type_shape = list(
            2 for _ in self.ext_preds
        )  # 对于每个存在量词谓词，状态有两种（真/假）。
        for idx, k in enumerate(self.cnt_params):  # 对于每个计数量词。
            if idx in self.mod_pred_index:  # 如果是模数类型。
                self.c_type_shape.append(k)  # 状态空间大小为 k (0 to k-1)。
            else:  # 如果是 =k或 <=k类型。
                self.c_type_shape.append(k + 1)  # 状态空间大小为 k+1 (0 to k)。

    def build_binary_evidence(self):
        """
        生成所有二元谓词在两个抽象元素 a, b 之间的所有可能真值指派组合。
        这些组合被称为“二元证据”，用于后续计算状态转移的权重。
        """
        ext_atoms = list(
            (
                (~pred(a, b), ~pred(b, a)),  # 组合1: pred(a,b)为假, pred(b,a)为假
                (~pred(a, b), pred(b, a)),  # 组合2: pred(a,b)为假, pred(b,a)为真
                (pred(a, b), ~pred(b, a)),  # 组合3: pred(a,b)为真, pred(b,a)为假
                (pred(a, b), pred(b, a)),  # 组合4: pred(a,b)为真, pred(b,a)为真
            )
            # 对 self.ext_preds 中的每一个二元谓词 pred 都生成这4种组合。
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
        ) # 对所有由二元计数量词引入的辅助谓词，执行完全相同的操作。
    
        # 使用 itertools.product 计算所有谓词的真值组合的笛卡尔积。这会生成一个迭代器，每次产生一个包含所有谓词真值情况的完整组合。例如，如果有2个谓词，每个有4种情况，这里就会产生 4*4=16 种最终组合。
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
        """解码最终结果，应用基数约束和重复因子调整。"""
        if not self.contain_cardinality_constraint(): # 如果没有基数约束，直接返回结果除以重复因子。
            res = res / self.repeat_factor
        else: # 如果存在基数约束，则先通过基数约束的symbolic weight解码方法处理结果，再除以重复因子。
            res = self.cardinality_constraint.decode_poly(
                res) / self.repeat_factor
        if self.leq_pred is not None: # 如果存在线性序公理，则乘以域大小的阶乘。
            res *= Rational(math.factorial(len(self.domain)), 1)
        return res

    def contain_cardinality_constraint(self) -> bool:
        return (
            self.cardinality_constraint is not None
            and not self.cardinality_constraint.empty()
        )

    def contain_linear_order_axiom(self) -> bool:
        """检查问题是否包含线性序公理。"""
        return self.problem.contain_linear_order_axiom()

    def build_t_update_dict(self, r, n_cells):
        """构建状态转移查找表 t_update_dict。
        构建一个巨大而详尽的状态转移查找表。
        该表存储了任意两个元素（处于任意可能的状态）在相互连接时，它们各自会如何
        转变到新状态，以及这次转变的权重是多少。
        通过提前计算好所有这些可能性，核心的递归算法在运行时就不需要动态计算状态转移，
        只需在此表中进行快速查找，从而极大地提升了性能.

        Args:
            r (dict): 在 build_weight中计算的关系字典。它存储了当 cell i 和 cell j
                      交互时，可能的状态 变化量 (dt, reverse_dt) 及其权重。
            n_cells (int): 单元格类型的总数。

        Returns:
            defaultdict: 一个嵌套的默认字典，结构为：
                         t_update_dict[(c1, c2)][(c1_new, c2_new)] = weight
                         其中 (c1, c2) 是配对前两个元素的完整状态，
                         (c1_new, c2_new) 是配对后的新状态。
        """
        # --- 初始化查找表
        t_update_dict = defaultdict(
            lambda: defaultdict(lambda: Rational(0, 1))
        ) # 创建一个嵌套的默认字典。结构为： t_update_dict[(c1, c2)][(c1_new, c2_new)] = weight。其中(c1, c2) 是配对前两个元素的状态，(c1_new, c2_new) 是配对后的新状态。

        # --- 预计算所有可能的c type状态组合, 下面的t相当于c type
        if self.exist_mod:
            final_list = [
                tuple(range(2)) for _ in self.ext_preds
            ]
            for idx, k in enumerate(self.cnt_params):
                if idx in self.mod_pred_index:
                    final_list += [tuple(range(k))]
                else:
                    final_list += [tuple(range(k + 1))]
            all_ts = list(product(*(final_list))) # 使用 itertools.product 计算 final_list中所有状态范围的笛卡尔积。这会生成所有可能的状态向量组合。# 例如，如果 final_list 是 [(0,1), (0,1,2)]，product会生成 (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)。
        else:
            all_ts = list(
                product(
                    *(
                        [tuple(range(2)) for _ in self.ext_preds]
                        + [tuple(range(i + 1)) for i in self.cnt_params]
                    )
                )
            )
        #
        # 遍历所有可能性并填充查找表
        for i in range(n_cells): # 遍历第一个元素的单元格类型 i。
            for j in range(n_cells): # 遍历第二个元素的单元格类型 j。
                for t1 in all_ts: # 遍历第一个元素的所有可能状态 t1。
                    for t2 in all_ts: # 遍历第二个元素的所有可能状态 t2。
                        for (dt, reverse_dt), rijt in r[(i, j)].items(): #  r是在 build_weight中计算的关系字典。它存储了当 cell i 和 cell j 交互时，可能的状态 *变化量* (dt, reverse_dt) 及其权重 rijt。
                            # --- 计算新状态
                            t1_new = [a - b for a, b in zip(t1, dt)] # 通过从原始状态减去变化量来计算新状态。
                            t2_new = [a - b for a, b in zip(t2, reverse_dt)]
                            # 
                            # --- 处理模数约束
                            if self.exist_mod:
                                for p, k_i in enumerate(self.cnt_params): # 遍历所有计数量词
                                    index = (len(self.ext_preds) + p) # 获取计数量词在c type中的索引位置 
                                    if p in self.mod_pred_index: # 如果这是一个模数约束
                                        t1_new[index] %= k_i 
                                        t2_new[index] %= k_i
                            # --- 剪枝：检查状态是否有效
                            if any(
                                t1_new[len(self.ext_preds) + p] < 0
                                or t2_new[len(self.ext_preds) + p] < 0
                                for p in range(len(self.cnt_params))
                            ): # 检查普通计数约束的状态是否变为负数。
                                continue  # 如果状态变为负数，说明这是一个无效的转移，跳过。
                            # --- 处理存在量词
                            for idx in range(
                                len(self.ext_preds)
                            ):
                                t1_new[idx] = max(t1_new[idx], 0) # 保持它们不为负数。
                                t2_new[idx] = max(t2_new[idx], 0)
                            # --- 组装完整的c type
                            c1 = (i,) + t1
                            c2 = (j,) + t2
                            c1_new = (i,) + tuple(t1_new)
                            c2_new = (j,) + tuple(t2_new)
                            # --- 更新查找表
                            t_update_dict[(c1, c2)][
                                (c1_new, c2_new)
                            ] += rijt
        return t_update_dict

    def build_weight(self, cells, cell_graph):
        """
        构建权重和关系字典 r
        该函数根据给定的单元格和单元格图计算权重字典和关系字典，
        用于后续的状态转移计算
        Args:
            cells: 单元格类型列表，一个列表，包含所有Cell类型
            cell_graph: 一个对象，预计算了Cell和Cell对的权重
        Returns:
            tuple: (w2t 字典, w 权重字典, r 关系字典)
                   w2t: 从单元格索引到谓词状态字典的字典, 例如 {0: (1, 0, 1, 2), 1: (0, 1, 0, 1)}, 值代表需要满足的数量
                   w: 每个单元格类型的权重字典，例如 {0: Rational(1, 1), 1: Rational(2, 1)}, 键是单元格索引，值是权重
                   r: 单元格对之间的关系字典，
        """

        n_cells = len(cells)  # 获取Cell类型的总数。
        w2t = dict() # 初始化 w2t 字典，用于存储从单元格索引到其目标状态向量的映射。
        w = defaultdict(
            lambda: Rational(0, 1)
        )  # 初始化 w 字典，用于存储每个Cell的权重。
        r = defaultdict(
            lambda: defaultdict(lambda: Rational(0, 1))
        )  # 初始化 r 字典。这是一个嵌套字典，结构为 r[(i, j)][(t, reverse_t)] = weight，用于存储从Cell对 (i, j) 到特定状态转移 (t, reverse_t) 的权重。
        for i in range(n_cells):  # 遍历所有单元格，索引为 i
            cell_weight = cell_graph.get_cell_weight(cells[i])  # 获取当前单元格的权重
            logger.debug("Cell %d weight: %s", i, cell_weight)
            t = list()  # 初始化状态列表t，用于存储谓词状态 (1=true, 0=false)

            # 计算存在量词（Skolem谓词）的目标状态
            for (
                pred
            ) in self.ext_preds:
                if cells[i].is_positive(pred): # 检查对于类型 i 的元素，Skolem谓词 pred 是否为真。
                    t.append(0) # 为真, 表示已经满足了存在量词的要求，不需要额外的关系支持。
                else:
                    t.append(1)  
            logger.debug("Cell %d 存在量词状态: %s", i, t)
            #
            # 计数量词
            # 逻辑解释: 假设有一个约束 ∀X ∃_{=k} Y: B(X,Y)，它被转换为 ∀X ∃_{=k} Y: @aux(X,Y)。当我们考虑一个元素 d_i 时，我们需要确保与它相关的 @aux 关系正好有 k 个。如果 cells[i] 使得 @aux(d_i, d_i) 为真（is_positive），那么这个元素自身就满足了1个关系。因此，它需要从其他 N-1 个元素那里再获得 k-1 个关系。所以目标状态设为 param - 1。如果 @aux(d_i, d_i) 为假，它就需要从其他 N-1 个元素那里获得全部 k 个关系。所以目标状态设为 param。mod 约束的逻辑类似，只是在做减法后需要取模。
            for idx, (pred, param) in enumerate(zip(self.cnt_preds, self.cnt_params)): # 这里的逻辑是计算一个元素为了满足计数量词，还需要从其他元素那里获得多少个关系。
                if cells[i].is_positive(pred): # 如果对自身为真，它已经满足了1个关系。
                    if (
                        self.exist_mod and idx in self.mod_pred_index
                    ): # 对于模数约束，目标变为 (r-1)。
                        t.append(
                            self.cnt_remainder[idx] - 1
                        )
                    else: # 对于普通计数约束（如 =k），目标变为 (k-1)。
                        t.append(param - 1)
                else: # 如果对自身为假，它需要从其他元素那里获得全部所需的关系。
                    if (
                        self.exist_mod and idx in self.mod_pred_index
                    ): # 对于模数约束，目标为 r。
                        t.append(self.cnt_remainder[idx])
                    else: # 对于普通计数约束，目标为 k。
                        t.append(param)
            logger.debug("Cell %d 计数量词状态: %s", i, t)
            w2t[i] = tuple(t)  # 将计算出的目标状态向量 t 存入 w2t 字典。
            w[i] = w[i] + cell_weight # 将该单元格类型的权重累加到 w 字典中。
            #
            # 开始计算二元转移权重。
            for j in range(n_cells):  # 启动一个内层循环，遍历所有Cell类型 j，形成Cell对 (i, j)。
                cell1 = cells[i]
                cell2 = cells[j]
                for evi_idx, evidence in enumerate(
                    self.binary_evidence
                ): # 遍历之前在 build_binary_evidence 中生成的所有可能的“二元证据”。
                    # 初始化两个列表，t 和 reverse_t，用于表示这次交互导致的状态变化。
                    t = list()
                    reverse_t = (
                        list()
                    )
                    # 从 cell_graph 获取在给定 evidence 下，cell1 和 cell2 交互的权重。
                    two_table_weight = cell_graph.get_two_table_weight(
                        (cell1, cell2), evidence
                    )
                    # 如果权重为0，说明这种交互不可能发生，直接跳过。
                    if two_table_weight == Rational(
                        0, 1
                    ):
                        continue
                    
                    # 根据 evidence (一个真值组合) 来构建状态变化向量 t 和 reverse_t。
                    # 这里的位运算是一种高效的解码方式，用于从 evi_idx 中提取每个谓词的真值。
                    for pred_idx, pred in enumerate(
                        self.ext_preds + self.cnt_preds
                    ):
                        # 检查 evidence 中 pred(b,a) 的真值。
                        if (
                            evi_idx >> (2 * pred_idx)
                        ) & 1 == 1:
                            reverse_t.append(1)
                        else:
                            reverse_t.append(0)
                        # 检查 evidence 中 pred(a,b) 的真值。
                        if (
                            evi_idx >> (2 * pred_idx + 1)
                        ) & 1 == 1:
                            t.append(1)
                        else:
                            t.append(0)
                    r[(i, j)][
                        (tuple(t), tuple(reverse_t))
                    ] = two_table_weight  # 使用谓词状态组合存储关系权重
        return w2t, w, r  # 返回映射字典、权重字典和关系字典

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return default, default

    def check_unary_constraints(self, config, mask) -> tuple[bool, bool, bool]:
        """
        检查给定配置是否违反一元约束。
        Args:
            config (iterable): 当前配置的状态向量。
            mask (tuple): 由 build_unary_mask 生成的一元约束掩码三元组。
            
        Returns:
            tuple: 三个布尔值，分别表示是否违反了一元模数约束、一元等号约束和一元小于等于约束。
        """
        return (
            self.check_unary_mod_constraints(config, mask[0]),
            self.check_unary_eq_constraints(config, mask[1]),
            self.check_unary_le_constraints(config, mask[2]),
        )

    def check_unary_mod_constraints(self, config, unary_mod_mask) -> bool:
        for mask, r_mod, k_mod in unary_mod_mask:            
            # `np.fromiter(config, dtype=np.int32)`: 将 `config` 对象（可能是一个自定义的封装类）高效地转换为一个NumPy数组，以便进行数学运算。
            # `mask @ ...`: `@` 是NumPy中的矩阵乘法运算符。在这里，它执行的是掩码向量 `mask` 和配置向量 `config` 之间的“点积”。由于 `mask` 在满足谓词的位置为1，其他位置为0，这个点积的结果正是当前配置中满足该约束谓词的元素总数。
            config_total_unary_constraint = mask @ np.fromiter(
                config, dtype=np.int32
            )
            if config_total_unary_constraint % k_mod != r_mod: # 检查计算出的总数是否满足模数约束。
                return True # 如果不等于，说明这个约束被违反了。函数立即返回 True，表示“发现了违规”。
        return False

    def check_unary_eq_constraints(self, config, unary_eq_mask) -> bool:
        for mask, k_eq in unary_eq_mask:
            if (mask @ np.fromiter(config, dtype=np.int32)) != k_eq: # 检查计算出的总数是否等于指定的 k_eq。
                return True # 如果不等于，说明这个约束被违反了。函数立即返回 True，表示“发现了违规”。
        return False

    def check_unary_le_constraints(self, config, unary_le_masks) -> bool:
        vec = np.fromiter(config, dtype=np.int32)
        for mask, k_max in unary_le_masks: 
            if (mask @ vec) > k_max: # 检查计算出的总数是否小于等于指定的 k_max。
                return True # 如果大于，说明这个约束被违反了。函数立即返回 True，表示“发现了违规”。
        return False

    def build_unary_mask(self, cells) -> tuple[list, list, list]:
        """构建一元约束掩码。
        返回三种类型的一元约束掩码 三元组：
        1. 一元模数约束掩码列表
        2. 一元等号约束掩码列表
        3. 一元小于等于约束掩码列表
        """
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
        """构建一元模数约束掩码列表。
        为每个一元模数约束（如 ∃_{≡r (mod k)}x: P(x)）生成一个掩码向量。
        这个掩码可以快速计算出在任意配置下，满足谓词P的元素总数。
    
        cells (list): 所有单元格类型的列表。

        Returns:
            list: 一个掩码列表，每个元素是一个元组 (mask, r, k)，其中
                  mask 是一个NumPy数组，r 和 k 是模数约束的参数。
        """
        n_cells = len(cells) # 获取单元格类型的总数，用于确定掩码向量的长度。
        masks = []
        for pred, r, k in self.unary_mod_constraints: # 遍历在 _build方法中解析并存储的所有一元模数约束。每个约束是一个元组 (pred, r, k)。
            mask = np.fromiter(
                (
                    # 这是一个生成器表达式，它会遍历所有的单元格类型。
                    # 对于每个 cell，检查它是否满足当前约束的谓词 pred。
                    # 如果满足 (cell.is_positive(pred) 返回 True)，则生成 1，否则生成 0。
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
    """
    一个带记忆化功能的配置更新器，用于高效计算状态转移。
    Cache_H 是一个嵌套的两层字典：
        第一层字典:
            键 (Key): 一个元组 (target_c, other_c)，代表了要进行配对的两种元素的初始状态。
            值 (Value): 第二层字典。
        第二层字典:
            键 (Key): 一个整数 j，代表与 target_c 配对的 other_c 元素的数量。
            值 (Value): 一个 H 字典，也就是 target_c 与 j 个 other_c 配对的完整结果。
    例子：
    Cache_H = {
        # 缓存了 (类型A, 类型B) 的配对结果
        ( (7, 1), (10, 0) ): {
            1: { ... H_for_j_1 ... },  # A 与 1 个 B 配对的结果
            2: { ... H_for_j_2 ... },  # A 与 2 个 B 配对的结果
            3: { ... H_for_j_3 ... }   # A 与 3 个 B 配对的结果
        },
    }
           
    H 是一个字典，其结构如下：
        键 (Key): 一个元组 (target_c_new, H_config_new)
            target_c_new: 目标元素在配对后的新状态坐标（一个元组）。
            H_config_new: 一个 HashableArrayWrapper 对象。它内部的数组记录了所有被配对的 j 个 other_c 元素在配对后各自变成了什么新状态。
        值 (Value): 一个 Rational 对象，代表达到这个特定状态组合的累计权重。
        
    例子：
        H = {
        ( (7, 0, ), <HashableArrayWrapper for [[0,0],[3,0]]> ) : Rational(12, 1),
        }
    """
    def __init__(self, t_update_dict, c1_type_shape, Cache_H):
        self.t_update_dict = t_update_dict # 一个预先计算好的巨大查找表，存储了任意两个元素配对时的状态转移规则和权重。
        self.c1_type_shape = c1_type_shape
        self.Cache_H = Cache_H # 一个缓存字典，用于存储已经计算过的配置更新结果，避免重复计算，提高效率。

    def f(self, target_c, other_c, l): # 核心方法，计算 target_c 与 l 个 other_c 配对的结果。
        # --- 1. 智能缓存查找 ---
        if (target_c, other_c) in self.Cache_H: # 检查是否曾经计算过 target_c和 other_c类型的配对。
            H_sub_dict = self.Cache_H[
                (target_c, other_c)
            ] # 如果有，获取该配对类型下的子缓存。这个子缓存按配对数量 l存储结果。
            num_start = l # 从需要计算的数量 l开始，向前查找最近的已缓存结果。
            while (
                num_start not in H_sub_dict and num_start > 0
            ):
                num_start -= 1 # 循环结束后，num_start是我们能找到的、小于等于l的、已缓存的最大数量。
        else: # 如果从未计算过这种配对，则初始化一个空的子缓存。
            self.Cache_H[(target_c, other_c)] = dict()
            num_start = 0 # 并设置 num_start为 0，表示必须从头开始计算。
        
        # --- 2. 初始化计算起点 H ---
        if num_start == 0: # 如果 num_start 为 0，说明没有任何缓存可用，需要创建初始状态。
            H = dict()
            H_config = np.zeros(
                self.c1_type_shape, dtype=np.uint8
            )
            H_config = HashableArrayWrapper(H_config)
            H[(target_c, H_config)] = Rational(
                1, 1
            )
        else: # 如果找到了缓存，直接从缓存中加载 num_start对应的结果作为计算起点。
            H = self.Cache_H[(target_c, other_c)][num_start]
        
        # --- 3. 迭代计算 ---
        for j in range(num_start + 1, l + 1): # 这个循环从已缓存的 num_start 步之后开始，一步步计算直到达到需要的数量 l。
            H_new = defaultdict(
                lambda: Rational(0, 1)
            ) # H_new 用于存储本次迭代（即与第 j 个元素配对）产生的新状态。
            for (target_c_old, H_config_old), W in H.items(): # 遍历上一步（与 j-1 个元素配对）的所有结果状态。
                for (target_c_new, other_c_new), rij in self.t_update_dict[
                    (target_c_old, other_c) # t_update_dict 存着 target_c_old 与一个 other_c 配对的所有可能转移, rij 是这次转移的权重。
                ].items():
                    H_config_new = np.array(H_config_old.array)
                    H_config_new[other_c_new] += 1 # other_c_new 是 other_c 的新状态。在新状态索引对应的字典上，元素数量加1。
                    H_config_new = HashableArrayWrapper(
                        H_config_new
                    )
                    H_new[(target_c_new, H_config_new)] += (
                        W * rij
                    ) # 新权重 = 上一步的权重 W * 本次转移的权重 rij。
            H = H_new # 用本次计算出的新状态 H_new 覆盖 H，作为下一次迭代的起点。
            self.Cache_H[(target_c, other_c)][j] = H # 将本次迭代的结果（即与 j 个元素配对的结果）存入缓存，以备后用。
        return H


class HashableArrayWrapper(object):
    def __init__(self, input_array: np.ndarray):
        self.array = input_array.astype(np.uint8, copy=False) # 将传入的 NumPy 数组转换为 uint8 类型并存储。

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
