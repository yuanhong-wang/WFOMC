# 导入 Python 未来特性和标准库
from __future__ import annotations
from logzero import logger
from math import comb
from functools import reduce

# 导入构建公式所需的语法类
from wfomc.fol.syntax import *

# 导入项目内的辅助函数和基类
from wfomc.fol.utils import convert_counting_formula, new_predicate
from wfomc.network.constraint import CardinalityConstraint
from wfomc.utils.polynomial import Rational
from wfomc.context import WFOMCContext


class WFOMCContextNewEncoding(WFOMCContext):
    """
    WFOMC 上下文，实现了论文中描述的“新编码”方法。
    - 对于二元计数量词 (∀∃=k)，使用新编码以避免指数级复杂度。该方法通过引入“规范模型”的概念和特殊的权重来约束 Skolem 谓词，
    - 对于一元计数量词 (∃=k)，使用最高效的直接转换法（转为基数约束）。
    """

    def __init__(self, problem: WFOMCProblem, **kwargs):
        super().__init__(
            problem, **kwargs
        )  # 初始化方法，直接调用父类的初始化，行为与旧 Context 相同。

    def _build(self):
        """
        重写 _build 方法，以实现对不同计数量词的分别处理。
        """
        ## 步骤 1: 初始化，获取公式的无量词部分 (与旧方法相同)
        self.formula = self.sentence.uni_formula
        while not isinstance(self.formula, QFFormula):
            self.formula = self.formula.quantified_formula

        ext_formulas_to_skolemize = list(
            self.sentence.ext_formulas
        )  # 初始化待 Skolem 化的公式列表

        ## 步骤 2: 使用新编码处理计数量词 (核心修改部分)
        if self.sentence.contain_counting_quantifier():
            logger.info("Translating SC2 to SNF using the NEW encoding logic.")
            if not self.contain_cardinality_constraint():
                self.cardinality_constraint = CardinalityConstraint()

            for (
                cnt_formula
            ) in self.sentence.cnt_formulas:  # 遍历语句中所有的计数量词公式
                ## 2.1 【判断类型】检查计数量词是一元还是二元
                is_unary_counting = not isinstance(
                    cnt_formula.quantified_formula, QuantifiedFormula
                )  # 如果内部不是 QuantifiedFormula，则为一元

                ## 2.2 【情况 A】如果是一元计数 (例如 ∃=k X: P(X))
                if is_unary_counting:
                    logger.info(
                        f"Handling unary counting formula (direct conversion): {cnt_formula}"
                    )
                    # 直接调用 convert_counting_formula，它会高效地将其转为基数约束
                    uni_formula, ext_formulas, card_constraint, repeat_factor = (
                        convert_counting_formula(cnt_formula, self.domain)
                    )

                    # 将返回的基数约束添加到全局约束中 # 注意：uni_formula 会是 top, ext_formulas 是 [], repeat_factor 是 1
                    self.cardinality_constraint.add_simple_constraint(*card_constraint)

                ## 2.3 【情况 B】如果是二元计数 (例如 ∀X ∃=k Y: R(X,Y))
                else:
                    logger.info(
                        f"Handling binary counting formula with NEW encoding: {cnt_formula}"
                    )

                    ## 2.3.1 调用旧的转换函数，获取旧编码 Γ* 的所有组件
                    (
                        uni_formula_old,
                        ext_formulas_from_cnt,
                        card_constraint,
                        repeat_factor,
                    ) = convert_counting_formula(cnt_formula, self.domain)
                    # uni_formula_old: 包含 fᵢ 分解和互斥性约束的公式 # ext_formulas: k 个形如 ∀X∃Y:fᵢ(X,Y) 的公式，它们是 Aᵢ 谓词的“原料” # card_constraint: 基数约束，如 |R|=n*k # repeat_factor: 旧编码的修正因子 (k!)^n

                    k = (
                        cnt_formula.quantified_formula.quantifier_scope.count_param
                    )  # 提取计数参数 k，并提前创建新编码所需的谓词

                    ## 2.3.2 立即处理与该计数量词相关的 k 个存在量词公式
                    skolem_preds = []  # 用于存储 k 个 Aᵢ 谓词
                    skolem_axioms = top  # 用于收集 k 个 Skolem 化公理
                    for (
                        ext_formula
                    ) in ext_formulas_from_cnt:  # 遍历 k 个形如 ∀X∃Y:fᵢ(X,Y) 的公式
                        inner_formula = (
                            ext_formula.quantified_formula.quantified_formula
                        )  # 提取 fᵢ(X,Y) 部分
                        skolem_pred = new_predicate(
                            1, SKOLEM_PRED_NAME
                        )  # 为当前的 fᵢ 创建一个唯一的 Skolem 谓词 (Aᵢ)
                        skolem_preds.append(skolem_pred)  # 保存 Aᵢ 以便后续使用
                        X = Var("X")
                        axiom_body = (
                            skolem_pred(X) | ~inner_formula
                        )  # 生成 Skolem 化公理: Aᵢ(X) ∨ ¬fᵢ(X,Y)
                        skolem_axioms &= axiom_body  # 将公理加入集合
                        self.weights[skolem_pred] = (
                            Rational(1, 1),
                            Rational(-1, 1),
                        )  # 为新创建的 Aᵢ 设置 (1, -1) 的权重

                    ## 2.3.3 【引入 Cᵢ】创建 k+1 个新的 "规范状态" 标记谓词 Cᵢ
                    c_preds = [new_predicate(1, f"C_{j}_") for j in range(k + 1)]

                    ## 2.3.4 【构建 Γ꜀ 约束】精确定义每个 Cⱼ 与 Aᵢ 组合的等价关系
                    gamma_c_body = top
                    X = Var("X")
                    for j in range(k + 1):
                        # 构建等价关系(⇔)的右侧：前 j 个 Aᵢ 为真，其余 k-j 个 Aᵢ 为假 # 使用 reduce 和 & 操作符构建合取链，代码更简洁
                        true_atoms = [skolem_preds[h](X) for h in range(j)]
                        false_atoms = [~skolem_preds[h](X) for h in range(j, k)]
                        conj_true_A = reduce(lambda f1, f2: f1 & f2, true_atoms, top)
                        conj_false_A = reduce(lambda f1, f2: f1 & f2, false_atoms, top)
                        definition = conj_true_A & conj_false_A
                        gamma_c_body &= c_preds[j](X).equivalent(
                            definition
                        )  # 将 Cⱼ(X) ⇔ definition 这条规则加入到总约束中

                    ## 2.3.5 【构建强制析取】构建 ∀x ⋁ Cⱼ(x) 规则，强制模型必须是规范的
                    disjuncts = [
                        p(X) for p in c_preds
                    ]  # 使用 reduce 和 | 操作符来构建析取。 这部分实现了论文中的公式: ∀x ⋁ Cⱼ(x)
                    final_disjunction_body = reduce(
                        lambda f1, f2: f1 | f2, disjuncts, bot
                    )

                    ## 2.3.6 【组合】将所有部分组合成最终的无量词公式
                    self.formula &= (
                        uni_formula_old
                        & skolem_axioms
                        & gamma_c_body
                        & final_disjunction_body
                    )

                    ## 2.3.7 【加权】为 Cᵢ 设置特殊的“魔法权重” w(Cⱼ) = C(k, j)
                    for j in range(k + 1):
                        weight = Rational(comb(k, j), 1)
                        neg_weight = Rational(1, 1)
                        self.weights[c_preds[j]] = (weight, neg_weight)

                    ## 2.3.8 【更新】更新全局的基数约束和修正因子
                    self.cardinality_constraint.add_simple_constraint(*card_constraint)
                    self.repeat_factor *= repeat_factor

        ## --- 步骤 3: 后续处理 (与之前相同) ---
        if self.contain_cardinality_constraint():
            self.cardinality_constraint.build()

        # 只处理与计数量词无关的原始存在量词公式
        for ext_formula in ext_formulas_to_skolemize:
            self.formula &= self._skolemize_one_formula(ext_formula)

        if self.contain_cardinality_constraint():
            self.weights.update(
                self.cardinality_constraint.transform_weighting(self.get_weight)
            )

        if self.problem.contain_linear_order_axiom():
            self.leq_pred = Pred("LEQ", 2)
