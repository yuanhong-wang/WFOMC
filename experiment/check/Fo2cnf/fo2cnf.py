import os
import sys
import argparse
import copy
import sympy
import re
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Set, Iterable, Tuple
from itertools import product, combinations
from logzero import logger, loglevel, logfile

from wfomc import parse_input
from wfomc.fol.sc2 import SC2
from wfomc.fol.syntax import Const, X, Y, QFFormula, AtomicFormula, Pred, top

from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.card import CardEnc


ganak_path = "/home/sunshixin/software/ganak/ganak"  # 如果ganak在linux服务器的PATH中
approxmc_path = "/home/sunshixin/software/approxmc/approxmc"  # 如果approxmc在PATH中


class CNFContext:
    def __init__(self, file_path, domain_size):
        self.file_path = file_path  # 输入文件路径
        path = Path(self.file_path)
        self.file_name = path.name  # 输入文件名
        self.file_dir = path.parent  # 输入文件目录
        self.problem = parse_input(self.file_path)  # 解析输入文件得到问题对象
        self.problem.domain = {Const(str(i)) for i in range(
            domain_size)}  # 根据自定义输入的domain来设置
        self.domain = self.problem.domain
        self.expr = sympy.true  # 最终的sympy表达式
        self.atom_to_id: Dict[AtomicFormula, int] = {}  # 原子到变量ID的映射
        self.sym_to_id: Dict[sympy.Symbol, int] = {}  # sympy符号到变量ID的映射
        self.next_var_id = 1  # 下一个可用的变量ID

        # 输出cnf文件名
        cnf_file_name = f"{os.path.splitext(self.file_name)[0]}_domain_size_{domain_size}.cnf"
        # 输出子句文件名
        clause_file_name = f"{os.path.splitext(self.file_name)[0]}_domain_size_{domain_size}.txt"
        self.cnf_path = os.path.join(self.file_dir, cnf_file_name)
        self.clause_path = os.path.join(self.file_dir, clause_file_name)
        self.clauses: list[list[int]] = []  # 存储CNF子句的列表

    def convert(self):
        """主转换流程"""
        os.makedirs(os.path.dirname(self.cnf_path),
                    exist_ok=True)  # 确保输出cnf目录存在

        # 1. 将所有sympy相关的公式转换为一个大的sympy表达式
        self._ground_universal()  # ground universal formulas
        self._ground_extension()  # ground extension formulas
        self._ground_counting()

        if self._check_exists_linear_order():
            self._add_linear_order_axioms()
        # 2. 将这个大的sympy表达式转换为CNF子句并添加到self.clauses
        self._add_clauses_from_expr()

        # 3. 独立处理基数约束，直接生成CNF子句并添加到self.clauses
        if self.problem.cardinality_constraint and not self.problem.cardinality_constraint.empty():
            self._ground_cardinality_constraints()

    def _register_atom(self, atom: AtomicFormula):
        """注册一个原子，如果不存在则分配新ID。"""
        if atom not in self.atom_to_id:
            self.atom_to_id[atom] = self.next_var_id  # 注册atom对象
            self.sym_to_id[atom.expr] = self.next_var_id  # 注册对应的sympy符号
            self.next_var_id += 1  # 更新下一个可用变量ID

    def _add_clauses_from_expr(self):
        """将Sympy表达式转换为CNF子句并添加到self.clauses中。"""
        if self.expr is sympy.true:
            return
        if self.expr is sympy.false:
            self.clauses.append([])  # 添加空子句表示不可满足
            return

        cnf_expr = sympy.to_cnf(self.expr)
        expr_clauses = cnf_expr.args if isinstance(
            cnf_expr, sympy.And) else [cnf_expr]

        for cl in expr_clauses:
            lits = []
            # 如果 cl 是一个 Symbol 或 Not，它就是一个单元子句
            if isinstance(cl, (sympy.Symbol, sympy.Not)):
                atoms = [cl]
            # 如果 cl 是一个 Or，它是一个多文字子句
            elif isinstance(cl, sympy.Or):
                atoms = cl.args
            else:
                # 可能是 True 或其他情况，直接跳过
                continue

            for at in atoms:
                if isinstance(at, sympy.Symbol):
                    if at in self.sym_to_id:
                        lits.append(self.sym_to_id[at])
                elif isinstance(at, sympy.Not):
                    base = at.args[0]
                    if base in self.sym_to_id:
                        lits.append(-self.sym_to_id[base])
            if lits:
                self.clauses.append(lits)

    def _extract_qf(self, formula):
        while not isinstance(formula, QFFormula):
            formula = formula.quantified_formula
        return formula

    def _ground_universal(self):
        uni_qf = self._extract_qf(copy.deepcopy(
            self.problem.sentence.uni_formula))  # 提取最内层的无量词公式

        # 使用 product 遍历所有有序对 (e1, e2)，确保对非对称公式的正确基化
        for e1, e2 in product(self.domain, repeat=2):
            grounded = uni_qf.substitute(
                {X: e1, Y: e2})  # 进行替换，把谓词里面的变量符号替换为具体的常量
            if grounded is top:  # 如果替换结果为 True，则跳过
                continue
            if grounded.expr is None:  # 如果替换结果为 False，则整个公式不可满足

                return sympy.false

            for atom in grounded.atoms():  # 注册所有出现的原子
                self._register_atom(atom)
            self.expr &= grounded.expr  # 将已经ground后的表达式与当前表达式进行与操作

    def _ground_extension(self):
        ext_qfs = [self._extract_qf(copy.deepcopy(f.quantified_formula))
                   for f in self.problem.sentence.ext_formulas]  # 提取所有存在Formula的最内层无量词公式，形成一个列表
        if not ext_qfs:  # 如果没有存在公式，直接返回
            return
        for e1 in self.domain:
            for ext_qf in ext_qfs:
                disjunction = sympy.false
                for e2 in self.domain:
                    grounded = ext_qf.substitute({X: e1, Y: e2})
                    for atom in grounded.atoms():
                        self._register_atom(atom)
                    disjunction |= grounded.expr  # 将所有ground formulas进行或操作, 也就是存在量词的语义
                self.expr &= disjunction  # 将存在量词的结果与当前表达式expr进行与操作
                logger.debug(
                    f"Expression after grounding extension formulas: {disjunction}")

    def _ground_cardinality_constraints(self):
        """直接生成基数约束的CNF子句并添加到self.clauses中。"""
        for pred_map, op, bound in self.problem.cardinality_constraint.constraints: # ({Eq: 1.0}, '=', 5.0)
            for pred, _ in pred_map.items():
                pred_name = str(pred)
                k = int(bound)

                related_vars = [
                    v for ksym, v in self.sym_to_id.items() if pred_name in str(ksym)] # 获取所有与该谓词相关的变量ID。related_vars = [3, 8, 13, 19, 23, 25, 28, 33, 
                if not related_vars: # 没有相关变量，跳过
                    continue

                # 使用pysat生成CNF子句
                if op == "<=":
                    cnf_cc = CardEnc.atmost(
                        lits=related_vars, bound=k, top_id=self.next_var_id - 1)
                elif op == ">=":
                    cnf_cc = CardEnc.atleast(
                        lits=related_vars, bound=k, top_id=self.next_var_id - 1)
                elif op == "=":
                    cnf_cc = CardEnc.equals(
                        lits=related_vars, bound=k, top_id=self.next_var_id - 1) # # top_id=4 表示新引入的辅助变量将从5开始
                else:
                    raise RuntimeError(f"Unknown operator: {op}")

                # 直接将生成的子句添加到self.clauses，并更新变量计数器
                self.clauses.extend(cnf_cc.clauses)
                self.next_var_id = cnf_cc.nv + 1 # cnf_cc.nv 是编码后的总变量数目，更新下一个可用变量ID

    def _is_single_layer(self, formula):
        """检查计数公式是否为单层计数公式。"""
        return isinstance(formula.quantified_formula, QFFormula)  # 只需要判断内存是否是QFFormula类型即可

    def _ground_counting(self):
        cnt_formulas = self.problem.sentence.cnt_formulas  # 提取所有计数公式
        single_layer_formulas, double_layer_formulas = [], []  # 分别存储单层和双层计数公式
        for f in cnt_formulas:
            (single_layer_formulas if self._is_single_layer(
                f) else double_layer_formulas).append(f)  # 分类存储

        for formula in single_layer_formulas:
            scope = formula.quantifier_scope  # 提取量词作用域
            if scope.comparator not in ('=', 'mod'):  # 检查比较器是否合法
                raise RuntimeError(
                    f"Unsupported comparator in counting quantifier: {scope.comparator}")
            if scope.comparator == "=":
                self.expr &= self._build_eq(
                    formula.quantified_formula, None, scope.quantified_var, None, scope.count_param)
            elif scope.comparator == "mod":
                self.expr &= self._build_mod(
                    formula.quantified_formula, None, scope.quantified_var, None, scope.count_param)

        for e1 in self.domain:
            for formula in double_layer_formulas:
                scope = formula.quantified_formula.quantifier_scope  # 内层scope
                qf = formula.quantified_formula.quantified_formula  # 内层QFFormula
                free_vars = qf.vars() - {scope.quantified_var}
                # 获取外层的变量X，也就是相对inner formular自由的X
                var_x = next(iter(free_vars)) if free_vars else None
                if scope.comparator not in ('=', 'mod'):
                    raise RuntimeError(
                        f"Unsupported comparator in counting quantifier: {scope.comparator}")
                if scope.comparator == "=":  # 内层等于=
                    self.expr &= self._build_eq(
                        qf, var_x, scope.quantified_var, e1, scope.count_param)  # 传入外层变量X和对应的常量
                elif scope.comparator == "mod":  # 内层等于mod
                    self.expr &= self._build_mod(
                        qf, var_x, scope.quantified_var, e1, scope.count_param)

    def _build_eq(self, inner_qf, var_x, var_y, e1, k):
        """构建等于k的计数子句表达式。inner_qf: 内层QF公式，var_x: 外层变量符号，var_y: 内层变量符号，e1: 外层变量对应的常量，k: 计数值。"""
        clause_expr = sympy.false
        # 根据domain生成所有可能y的 k中组合
        for y_const_list in combinations(self.domain, k):
            sub_conj = sympy.true
            domain_set, y_const = set(self.domain), set(
                y_const_list)  # 将domain和备选组合转换为集合以便后续操作
            for y in y_const:  # 遍历所有可选的y常量
                # 双层，有var_X, 那么{Y: Const('1'), X: Const('0')}
                # 单层，无var_X, 那么{Y: Const('1')}
                subst = {var_y: y, **({var_x: e1} if var_x else {})}  # 构建替换字典,
                g = inner_qf.substitute(subst)
                for atom in g.atoms():
                    self._register_atom(atom)
                sub_conj &= g.expr
            for y in domain_set - y_const:  # 遍历所有不在备选组合中的y常量，没有选择这些y,对应的公式取反
                subst = {var_y: y, **({var_x: e1} if var_x else {})}
                g = inner_qf.substitute(subst)
                for atom in g.atoms():
                    self._register_atom(atom)
                sub_conj &= ~g.expr
            clause_expr |= sub_conj  # 将所有满足k个的子句进行或操作，也就是存在这个情况，然后总共循环k次, 也就是k个存在的情况
        # 最后将整个表达式转换为CNF形式
        return sympy.simplify_logic(clause_expr, form='cnf')

    def _build_mod(self, inner_qf, var_x, var_y, e1, rk):
        """构建模k的计数子句表达式。inner_qf: 内层QF公式，var_x: 外层变量符号，var_y: 内层变量符号，e1: 外层变量对应的常量，rk: (r, k)元组。"""
        r, k_mod = rk
        clause_expr = sympy.false
        # 遍历所有满足模k条件 的y常量数量取值
        for possible_y_count in range(len(self.domain) + 1):
            if possible_y_count % k_mod != r:
                continue
            # 根据domain生成 这些数量y的 possible_y_count中组合
            for y_const_list in combinations(self.domain, possible_y_count):
                sub_conj = sympy.true  # 下面代码和上面的_build_eq类似
                domain_set, y_const = set(self.domain), set(y_const_list)
                for y in y_const:
                    subst = {var_y: y, **({var_x: e1} if var_x else {})}
                    g = inner_qf.substitute(subst)
                    for atom in g.atoms():
                        self._register_atom(atom)
                    sub_conj &= g.expr
                for y in domain_set - y_const:
                    subst = {var_y: y, **({var_x: e1} if var_x else {})}
                    g = inner_qf.substitute(subst)
                    for atom in g.atoms():
                        self._register_atom(atom)
                    sub_conj &= ~g.expr
                clause_expr |= sub_conj
        return sympy.simplify_logic(clause_expr, form='cnf')

    def dump(self):
        """将 self.clauses 中的所有子句写入CNF文件。"""
        num_vars = self.next_var_id - 1
        num_clauses = len(self.clauses)
        with open(self.cnf_path, 'w', encoding='utf-8') as f:
            f.write(f"p cnf {num_vars} {num_clauses}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")

        logger.info(
            f"CNF file with {num_vars} vars and {num_clauses} clauses written to: {self.cnf_path}")

    def _check_exists_linear_order(self) -> bool:
        """检查问题中是否存在LEQ谓词以决定是否进行线性序编码。"""
        if "LEQ" in [pred.name for pred in self._collect_all_predicates()]:  # 遍历所有谓词列表，查找LEQ谓词
            logger.info("Linear order predicate detected")
            return True
        return False

    def _add_linear_order_axioms(self) -> sympy.Expr:
        """
        生成线性序的公理 (Reflexivity, Antisymmetry, Transitivity, Totality)
        并返回一个 sympy 表达式。
        这允许求解器探索所有可能的 n! 种排序。
        """
        logger.debug("Adding axioms for linear order predicate (LEQ).")

        axioms_expr = sympy.true
        leq_pred = Pred('LEQ', 2)

        # 预先注册所有可能的 LEQ 原子，确保它们都有变量ID
        for c1, c2 in product(self.domain, repeat=2):
            self._register_atom(AtomicFormula(leq_pred, (c1, c2), True))

        # 添加公理
        # 自反性: forall x. LEQ(x,x) 。遍历domain中的每一个元素 x，生成表达式 LEQ(x, x)，并将它们用逻辑与（&）连接起来。
        for x in self.domain:
            axioms_expr &= AtomicFormula(leq_pred, (x, x), True).expr

        # 反对称性: forall x,y. (LEQ(x,y) & LEQ(y,x)) -> x=y
        #    Equivalent to: forall x!=y. ~LEQ(x,y) | ~LEQ(y,x)
        """
        逻辑定义: ∀x, y: (LEQ(x, y) & LEQ(y, x)) -> (x = y)
        等价形式: ∀x ≠ y: ¬LEQ(x, y) ∨ ¬LEQ(y, x)
        通俗解释: 如果 x 小于等于 y，并且 y 也小于等于 x，那么 x 和 y 必然是同一个元素。换句话说，对于两个不同的元素，它们之间的小于等于关系只能是单向的。
        """
        for x, y in combinations(self.domain, 2):
            axioms_expr &= sympy.Or(
                AtomicFormula(leq_pred, (x, y), False).expr,
                AtomicFormula(leq_pred, (y, x), False).expr
            )

        # 3.3 传递性: forall x,y,z. (LEQ(x,y) & LEQ(y,z)) -> LEQ(x,z)
        #    Equivalent to: ~LEQ(x,y) | ~LEQ(y,z) | LEQ(x,z)
        """
        逻辑定义: ∀x, y, z: (LEQ(x, y) & LEQ(y, z)) -> LEQ(x, z)
        等价形式: ∀x, y, z: ¬LEQ(x, y) ∨ ¬LEQ(y, z) ∨ LEQ(x, z)
        通俗解释: 如果 x 小于等于 y，y 小于等于 z，那么 x 一定小于等于 z。
        """
        for x, y, z in product(self.domain, repeat=3):
            axioms_expr &= sympy.Or(
                AtomicFormula(leq_pred, (x, y), False).expr,
                AtomicFormula(leq_pred, (y, z), False).expr,
                AtomicFormula(leq_pred, (x, z), True).expr
            )

        # 3.4  完全性: forall x,y. LEQ(x,y) | LEQ(y,x)
        """
        对于任意两个元素 x 和 y，要么 x 小于等于 y，要么 y 小于等于 x（或者两者都成立，如果 x=y）。它们之间总是“可比较的”。
        """
        for x, y in product(self.domain, repeat=2):
            axioms_expr &= sympy.Or(
                AtomicFormula(leq_pred, (x, y), True).expr,
                AtomicFormula(leq_pred, (y, x), True).expr
            )
        self.expr &= axioms_expr

    def _collect_all_predicates(self) -> Set[Pred]:
        """
        遍历问题中的所有公式，收集所有唯一的谓词对象。
        """
        all_preds = set()  # 用于存储所有唯一的谓词对象
        formulas_to_check = []  # 存储需要检查的公式列表

        # 从各种公式中收集
        if self.problem.sentence.uni_formula:
            formulas_to_check.append(self.problem.sentence.uni_formula)
        formulas_to_check.extend(self.problem.sentence.ext_formulas)
        formulas_to_check.extend(self.problem.sentence.cnt_formulas)

        for formula in formulas_to_check:
            # 递归地深入到最内层的无量词公式(QFFormula)
            qf_formula = self._extract_qf(copy.deepcopy(formula))
            for atom in qf_formula.atoms():
                all_preds.add(atom.pred)

        # 从基数约束中收集
        if self.problem.cardinality_constraint and not self.problem.cardinality_constraint.empty():
            for pred_map, _, _ in self.problem.cardinality_constraint.constraints:
                for pred in pred_map.keys():
                    all_preds.add(pred)

        # 从一元证据中收集 (如果存在)
        if self.problem.unary_evidence:
            for pred in self.problem.unary_evidence.keys():
                all_preds.add(pred)

        return all_preds

    # 定义类方法是为了方便单独调用
    @staticmethod
    def model_count_pysat(cnf_path: str) -> int:
        cnf = CNF(from_file=cnf_path)
        count = 0
        with Solver(bootstrap_with=cnf) as s:
            for _ in s.enum_models():
                count += 1
        return count

    @staticmethod
    def model_count_ganak(cnf_path: str) -> int:
        result = subprocess.run([ganak_path, cnf_path],
                                capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Ganak execution failed: {result.stderr}")
            return 0

        match = re.search(r"(?:s mc|c s exact arb int)\s+(\d+)", result.stdout)
        if match:
            return int(match.group(1))

        logger.warning(f"Could not parse Ganak output: {result.stdout}")
        return 0

    @staticmethod
    def model_count_approxmc(cnf_path: str, epsilon, delta) -> dict:
        """
        使用ApproxMC进行近似模型计数
        """
        cmd = [
            approxmc_path,
            f"--epsilon={epsilon}",
            f"--delta={delta}",
            cnf_path
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            if line.startswith("s mc"):
                return int(line.split()[-1])  # 提取最后的数字
        raise RuntimeError("ApproxMC 未返回计数结果")

    def _handle_linear_order_hard_encoding(self):
        """
        硬编码 LEQ 谓词的真值。也就是假定0<1<2<...这唯一的一种排序方法，这个方法没有被使用
        这会为所有相关的基化原子生成单元子句。
        此方法假定所有需要的 LEQ 原子已在之前的步骤中被注册。
        """
        logger.debug("Applying hard encoding for LEQ predicate.")

        # 建立领域的规范顺序
        sorted_domain_consts = sorted(
            list(self.domain), key=lambda c: int(str(c)))

        # 遍历所有已经注册的原子，为 LEQ 原子生成单元子句
        # 不再手动注册新原子，只处理已经存在的
        for atom, var_id in self.atom_to_id.items():
            if atom.pred.name == 'LEQ':
                # 确保原子参数是两个
                if len(atom.args) != 2:
                    continue

                left, right = atom.args
                try:
                    li = sorted_domain_consts.index(left)
                    ri = sorted_domain_consts.index(right)
                except ValueError:
                    # 如果原子中的常量不在领域内，则跳过
                    continue

                # 根据顺序添加单元子句
                if li <= ri:
                    self.clauses.append([var_id])   # LEQ(left, right) is True
                else:
                    self.clauses.append([-var_id])  # LEQ(left, right) is False

    def exists_cnf_file(self) -> bool:
        """输入wfomc文件路径，检查对应的cnf文件是否存在。"""
        if os.path.exists(self.cnf_path):
            logger.info(
                f"CNF file already exists in: {self.cnf_path}, skip convert and dump, and count directly.")
            return self.cnf_path
        return None

    def __str__(self):
        pass

    __repr__ = __str__


def Fo2Counter(file_path, domain_size, counters, epsilon=0.01, delta=0.01):
    """
    执行转换和模型计数的通用函数。
    """
    if not os.path.exists(file_path):  # 检查输入文件是否存在
        logger.error(f"Input file does not exist: {file_path}")
        return

    log_path = os.path.join(Path(__file__).parent, "performance.log")

    for counter in counters:
        try:
            logger.info(
                f"Processing file: {file_path}, Domain: {domain_size}, Counter: {counter}...")
            #
            context = CNFContext(file_path, domain_size)  # 创建CNF上下文
            # 检查CNF文件是否存在,如果存在就直接使用cnf文件，就不需要转换和dump了，直接计数
            cnf_path = context.exists_cnf_file()
            if not cnf_path:
                context.convert()  # 执行转换
                context.dump()  # 将CNF写入文件

            if counter == "pysat":
                count = CNFContext.model_count_pysat(context.cnf_path)
            elif counter == "ganak":
                count = CNFContext.model_count_ganak(context.cnf_path)
            elif counter == "approxmc":
                count = context.model_count_approxmc(
                    context.cnf_path, epsilon, delta)
            else:
                raise ValueError(f"Unknown counter: {counter}")
            #
            logger.info(
                f"Result:\n InputFile: {file_path}\n Domain Size: {domain_size}\n Counter: {counter}\n Model Count: {count}\n")
        except Exception as e:
            logger.error(
                f"An error occurred for {file_path} with counter {counter}: {e}", exc_info=True)
            logger.info(
                f"An error occurred for counter {counter}. Check logs at {log_path}.\n")


def parse_and_run():
    """
    处理命令行参数并启动转换。
    """
    global ganak_path, approxmc_path
    parser = argparse.ArgumentParser(
        description='Convert a first-order logic sentence to CNF and count models.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input sentence file name (e.g., m-odd-degree-graph-cc.wfomcs)')
    parser.add_argument('--domain', '-d', type=int,
                        default=5, help='Domain size (e.g., 3)')
    parser.add_argument('--counter', '-c', type=str, default='ganak',
                        choices=['ganak', 'pysat', 'approxmc'], help='Model counter to use')
    parser.add_argument('--ganak_path', type=str,
                        help='Path to Ganak executable')
    parser.add_argument('--approxmc_path', type=str,
                        help='Path to ApproxMC executable')
    parser.add_argument('--epsilon', type=float,
                        default=0.01, help='Epsilon for ApproxMC')
    parser.add_argument('--delta', type=float, default=0.01,
                        help='Delta for ApproxMC')
    args = parser.parse_args()
    if args.ganak_path:
        ganak_path = args.ganak_path
    else:  # 没有输入，就检查一下，确保ganak_path存在
        if os.path.exists(ganak_path):
            logger.debug(f"Using default Ganak path: {ganak_path}")
        else:
            logger.error(f"Ganak path does not exist: {ganak_path}")
            sys.exit(1)
    if args.approxmc_path:
        approxmc_path = args.approxmc_path
    else:  # 没有输入，就检查一下，确保approxmc_path存在
        if os.path.exists(approxmc_path):
            logger.debug(f"Using default ApproxMC path: {approxmc_path}")
        else:
            logger.error(f"ApproxMC path does not exist: {approxmc_path}")
            sys.exit(1)

    Fo2Counter(args.input, args.domain, [args.counter])


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    # 检查是否有命令行参数。sys.argv[0]是脚本名，所以len > 1表示有参数。

    if len(sys.argv) > 1:
        # 命令行模式：解析参数并运行
        parse_and_run()
    else:
        # 直接运行模式（例如在VS Code中右键运行）：使用默认值
        # file_path = "2-regular-graph.wfomcs"
        # file_path = "head-middle-tail.wfomcs"
        file_path = "/home/sunshixin/pycharm_workspace/WFOMC/models/BA_CC.wfomcs" # 注意E的约束和domain相同
        # file_path = "/home/sunshixin/pycharm_workspace/WFOMC/models/permutation-no-fix.wfomcs"
        # file_path = "/home/sunshixin/pycharm_workspace/WFOMC/models/0mod2-regular-graph.wfomcs"

        domain = 5
        default_counters = ["ganak"]
        Fo2Counter(file_path, domain, default_counters)
