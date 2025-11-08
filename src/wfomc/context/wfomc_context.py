import math
from logzero import logger
from copy import deepcopy
from wfomc.fol.sc2 import SC2
from wfomc.fol.utils import new_predicate, convert_counting_formula
from functools import reduce
from math import comb
from wfomc.network import CardinalityConstraint, PartitionConstraint, UnaryEvidenceEncoding, organize_evidence, \
    unary_evidence_to_ccs, unary_evidence_to_pc
from wfomc.fol.syntax import *
from wfomc.problems import WFOMCProblem
from wfomc.utils import Rational, RingElement


class WFOMCContext(object):
    """
    Context for WFOMC algorithm
    """

    def __init__(self, problem: WFOMCProblem,
                 unary_evidence_encoding: UnaryEvidenceEncoding = UnaryEvidenceEncoding.CCS):
        self.problem = deepcopy(problem)
        self.domain: set[Const] = self.problem.domain
        self.sentence: SC2 = self.problem.sentence
        self.weights: dict[Pred, tuple[RingElement,
                                       RingElement]] = self.problem.weights
        self.cardinality_constraint: CardinalityConstraint = self.problem.cardinality_constraint
        self.repeat_factor = 1
        self.unary_evidence = self.problem.unary_evidence
        self.unary_evidence_encoding = unary_evidence_encoding

        logger.info('sentence: \n%s', self.sentence)
        logger.info('domain: \n%s', self.domain)
        logger.info('weights:')
        for pred, w in self.weights.items():
            logger.info('%s: %s', pred, w)
        logger.info('cardinality constraint: %s', self.cardinality_constraint)
        logger.info('unary evidence: %s', self.unary_evidence)

        self.formula: QFFormula
        # for handling linear order axiom
        self.leq_pred: Pred = None
        # for handling unary evidence
        self.element2evidence: dict[Const, set[AtomicFormula]] = dict()
        self.partition_constraint: PartitionConstraint = None
        self._build()
        logger.info('Skolemized formula for WFOMC: \n%s', self.formula)
        logger.info('weights for WFOMC: \n%s', self.weights)
        logger.info('repeat factor: %d', self.repeat_factor)
        logger.info('partition constraint: %s', self.partition_constraint)

    def contain_cardinality_constraint(self) -> bool:
        return self.cardinality_constraint is not None and \
            not self.cardinality_constraint.empty()

    def contain_partition_constraint(self) -> bool:
        return self.partition_constraint is not None

    def contain_existential_quantifier(self) -> bool:
        return self.sentence.contain_existential_quantifier()

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return (default, default)

    def decode_result(self, res: RingElement) -> Rational:
        if not self.contain_cardinality_constraint():
            res = res / self.repeat_factor
        else:
            res = self.cardinality_constraint.decode_poly(
                res) / self.repeat_factor
        if self.leq_pred is not None:
            res *= Rational(math.factorial(len(self.domain)), 1)
        return res

    def _skolemize_one_formula(self, formula: QuantifiedFormula) -> QFFormula:
        """
        Only need to deal with \forall X \exists Y: f(X,Y) or \exists X: f(X,Y)
        """
        quantified_formula = formula.quantified_formula
        quantifier_num = 1
        while (not isinstance(quantified_formula, QFFormula)):
            quantified_formula = quantified_formula.quantified_formula
            quantifier_num += 1

        formula: QFFormula = top
        ext_formula = quantified_formula
        if not isinstance(ext_formula, AtomicFormula):
            aux_pred = new_predicate(quantifier_num, AUXILIARY_PRED_NAME)
            aux_atom = aux_pred(X, Y) if quantifier_num == 2 else aux_pred(X)
            formula = formula & (ext_formula.equivalent(aux_atom))
            ext_formula = aux_atom

        if quantifier_num == 2:
            skolem_pred = new_predicate(1, SKOLEM_PRED_NAME)
            skolem_atom = skolem_pred(X)
        elif quantifier_num == 1:
            skolem_pred = new_predicate(0, SKOLEM_PRED_NAME)
            skolem_atom = skolem_pred()
        formula = formula & (skolem_atom | ~ext_formula)
        self.weights[skolem_pred] = (Rational(1, 1), Rational(-1, 1))
        return formula

    def _skolemize(self) -> QFFormula:
        """
        Skolemize the sentence
        """
        formula = self.sentence.uni_formula
        while (not isinstance(formula, QFFormula)):
            formula = formula.quantified_formula
        for ext_formula in self.sentence.ext_formulas:
            formula = formula & self._skolemize_one_formula(ext_formula)
        return formula

    def _build(self):
        """
        重写 _build 方法，以实现对不同计数量词的分别处理。
        """
        # 步骤 1: 初始化，获取公式的无量词部分 (与旧方法相同)
        self.formula = self.sentence.uni_formula
        while not isinstance(self.formula, QFFormula):
            self.formula = self.formula.quantified_formula

        ext_formulas_to_skolemize = list(
            self.sentence.ext_formulas
        )  # 初始化待 Skolem 化的公式列表

        # 步骤 2: 使用新编码处理计数量词 (核心修改部分)
        if self.sentence.contain_counting_quantifier():
            logger.info("Translating SC2 to SNF using the NEW encoding logic.")
            if not self.contain_cardinality_constraint():
                self.cardinality_constraint = CardinalityConstraint()

            for (
                cnt_formula
            ) in self.sentence.cnt_formulas:  # 遍历语句中所有的计数量词公式
                # 2.1 【判断类型】检查计数量词是一元还是二元
                is_unary_counting = not isinstance(
                    cnt_formula.quantified_formula, QuantifiedFormula
                )  # 如果内部不是 QuantifiedFormula，则为一元

                # 2.2 【情况 A】如果是一元计数 (例如 ∃=k X: P(X))
                if is_unary_counting:
                    logger.info(
                        f"Handling unary counting formula (direct conversion): {cnt_formula}"
                    )
                    # 直接调用 convert_counting_formula，它会高效地将其转为基数约束
                    uni_formula, ext_formulas, card_constraint, repeat_factor = (
                        convert_counting_formula(cnt_formula, self.domain)
                    )

                    # 将返回的基数约束添加到全局约束中 # 注意：uni_formula 会是 top, ext_formulas 是 [], repeat_factor 是 1
                    self.cardinality_constraint.add_simple_constraint(
                        *card_constraint)

                # 2.3 【情况 B】如果是二元计数 (例如 ∀X ∃=k Y: R(X,Y))
                else:
                    logger.info(
                        f"Handling binary counting formula with NEW encoding: {cnt_formula}"
                    )

                    # 2.3.1 调用旧的转换函数，获取旧编码 Γ* 的所有组件
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

                    # 2.3.2 立即处理与该计数量词相关的 k 个存在量词公式
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

                    # 2.3.3 【引入 Cᵢ】创建 k+1 个新的 "规范状态" 标记谓词 Cᵢ
                    c_preds = [new_predicate(1, f"C_{j}_")
                               for j in range(k + 1)]

                    # 2.3.4 【构建 Γ꜀ 约束】精确定义每个 Cⱼ 与 Aᵢ 组合的等价关系
                    gamma_c_body = top
                    X = Var("X")
                    for j in range(k + 1):
                        # 构建等价关系(⇔)的右侧：前 j 个 Aᵢ 为真，其余 k-j 个 Aᵢ 为假 # 使用 reduce 和 & 操作符构建合取链，代码更简洁
                        true_atoms = [skolem_preds[h](X) for h in range(j)]
                        false_atoms = [~skolem_preds[h]
                                       (X) for h in range(j, k)]
                        conj_true_A = reduce(
                            lambda f1, f2: f1 & f2, true_atoms, top)
                        conj_false_A = reduce(
                            lambda f1, f2: f1 & f2, false_atoms, top)
                        definition = conj_true_A & conj_false_A
                        gamma_c_body &= c_preds[j](X).equivalent(
                            definition
                        )  # 将 Cⱼ(X) ⇔ definition 这条规则加入到总约束中

                    # 2.3.5 【构建强制析取】构建 ∀x ⋁ Cⱼ(x) 规则，强制模型必须是规范的
                    disjuncts = [
                        p(X) for p in c_preds
                    ]  # 使用 reduce 和 | 操作符来构建析取。 这部分实现了论文中的公式: ∀x ⋁ Cⱼ(x)
                    final_disjunction_body = reduce(
                        lambda f1, f2: f1 | f2, disjuncts, bot
                    )

                    # 2.3.6 【组合】将所有部分组合成最终的无量词公式
                    self.formula &= (
                        uni_formula_old
                        & skolem_axioms
                        & gamma_c_body
                        & final_disjunction_body
                    )

                    # 2.3.7 【加权】为 Cᵢ 设置特殊的“魔法权重” w(Cⱼ) = C(k, j)
                    for j in range(k + 1):
                        weight = Rational(comb(k, j), 1)
                        neg_weight = Rational(1, 1)
                        self.weights[c_preds[j]] = (weight, neg_weight)

                    # 2.3.8 【更新】更新全局的基数约束和修正因子
                    self.cardinality_constraint.add_simple_constraint(
                        *card_constraint)
                    self.repeat_factor *= repeat_factor

        if self.contain_cardinality_constraint():
            self.cardinality_constraint.build()

        # 只处理与计数量词无关的原始存在量词公式
        for ext_formula in ext_formulas_to_skolemize:
            self.formula &= self._skolemize_one_formula(ext_formula)

        # self.formula = self.formula.simplify()

        if self.contain_cardinality_constraint():
            self.weights.update(
                self.cardinality_constraint.transform_weighting(
                    self.get_weight,
                )
            )

        if self.problem.contain_linear_order_axiom():
            self.leq_pred = Pred('LEQ', 2)
