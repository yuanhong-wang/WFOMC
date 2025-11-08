import os
import copy
import argparse
from pathlib import Path
from typing import Dict, Set, Iterable, Tuple
import sympy
from itertools import product, combinations
from logzero import logger, loglevel
from wfomc import parse_input
from wfomc.problems import WFOMCProblem
from wfomc.fol.syntax import Const, X, Y, QFFormula, AtomicFormula, top
import subprocess
from itertools import combinations
import subprocess
from logzero import logger
from pysat.formula import CNF
from pysat.solvers import Solver
import pandas as pd
import re
import os
import copy
import sympy
from itertools import product, combinations
from pysat.formula import CNF
from pysat.solvers import Solver
from wfomc import parse_input
from wfomc.problems import WFOMCProblem
from wfomc.fol.syntax import Const, X, Y, QFFormula, top
import subprocess
from logzero import logger, logfile
import logging
from itertools import product, combinations, combinations_with_replacement
from pysat.card import CardEnc, EncType
from wfomc.fol.syntax import Const, X, Y, QFFormula, AtomicFormula, Pred

MODEL_DIR_PATH = "/home/sunshixin/pycharm_workspace/WFOMC/models"  # 这是输入文件的路径，需要根据实际路径修改
DIR_PATH = Path(__file__).parent  # 这是保存中间结果CNF输出结果的路径，然后后续进行操作
CNF_RESULTS_PATH = os.path.join(DIR_PATH, "cnf_results")  # 存储实验结果的路径
ganak_path = "/home/sunshixin/software/ganak/ganak"  # 如果ganak在linux服务器的PATH中
approxmc_path = "/home/sunshixin/software/approxmc/approxmc"  # 如果approxmc在PATH中


class CNFContext:
    def __init__(self, file_name, domain_size):
        self.input_path = os.path.join(MODEL_DIR_PATH, file_name)
        self.problem = parse_input(self.input_path)
        self.problem.domain = {Const(str(i)) for i in range(domain_size)}
        self.domain = self.problem.domain

        self.atom_to_id: Dict[AtomicFormula, int] = {}
        self.sym_to_id: Dict[sympy.Symbol, int] = {}
        self.next_var_id = 1
        
        cnf_file_name = f"{os.path.splitext(file_name)[0]}_domain_size_{domain_size}.cnf"
        self.cnf_path = os.path.join(CNF_RESULTS_PATH, cnf_file_name)
        self.clauses: list[list[int]] = []

    def convert(self):
        """主转换流程"""
        os.makedirs(os.path.dirname(self.cnf_path), exist_ok=True)

        # 1. 将所有sympy相关的公式转换为一个大的sympy表达式
        expr = sympy.true
        expr = self._ground_universal(expr)
        expr = self._ground_extensions(expr)
        expr = self.build_counting(self.problem.sentence.cnt_formulas, expr)

        # 2. 将这个大的sympy表达式转换为CNF子句并添加到self.clauses
        self._add_clauses_from_expr(expr)

        # 3. 独立处理基数约束，直接生成CNF子句并添加到self.clauses
        if self.problem.cardinality_constraint and not self.problem.cardinality_constraint.empty():
            self._ground_cardinality_constraints()

        # 4. 将所有收集到的子句写入文件
        self.dump()

    def _register_atom(self, atom: AtomicFormula):
        """注册一个原子，如果不存在则分配新ID。"""
        if atom not in self.atom_to_id:
            self.atom_to_id[atom] = self.next_var_id
            self.sym_to_id[atom.expr] = self.next_var_id
            self.next_var_id += 1

    def _add_clauses_from_expr(self, expr: sympy.Expr):
        """将Sympy表达式转换为CNF子句并添加到self.clauses中。"""
        if expr is sympy.true:
            return
        if expr is sympy.false:
            self.clauses.append([]) # 添加空子句表示不可满足
            return

        cnf_expr = sympy.to_cnf(expr)
        expr_clauses = cnf_expr.args if isinstance(cnf_expr, sympy.And) else [cnf_expr]
        
        for cl in expr_clauses:
            lits = []
            atoms = cl.args if isinstance(cl, sympy.Or) else [cl]
            for at in atoms:
                # 确保原子已经被注册
                if isinstance(at, sympy.Symbol):
                    if at in self.sym_to_id:
                        lits.append(self.sym_to_id[at])
                elif isinstance(at, sympy.Not):
                    base = at.args[0]
                    if base in self.sym_to_id:
                        lits.append(-self.sym_to_id[base])
            if lits:
                self.clauses.append(lits)

    def extract_qf(self, formula):
        while not isinstance(formula, QFFormula):
            formula = formula.quantified_formula
        return formula

    def _ground_universal(self, expr):
        uni_qf = self.extract_qf(copy.deepcopy(self.problem.sentence.uni_formula))
        for e1, e2 in combinations_with_replacement(self.domain, 2):
            grounded = uni_qf.substitute({X: e1, Y: e2}) & uni_qf.substitute({X: e2, Y: e1})
            if grounded is top:
                continue
            if grounded.expr is None:
                return sympy.false
            
            for atom in grounded.atoms():
                self._register_atom(atom)
            expr &= grounded.expr
            logger.info(f"\n{uni_qf}\n {'-' * 100} \n {grounded.expr}\n")
        return expr

    def _ground_extensions(self, expr):
        ext_qfs = [self.extract_qf(copy.deepcopy(f.quantified_formula)) for f in self.problem.sentence.ext_formulas]
        for e1 in self.domain:
            for ext_qf in ext_qfs:
                disjunction = sympy.false
                for e2 in self.domain:
                    grounded = ext_qf.substitute({X: e1, Y: e2})
                    for atom in grounded.atoms():
                        self._register_atom(atom)
                    disjunction |= grounded.expr
                expr &= disjunction
                logger.info(f"Expression after grounding extension formulas: {disjunction}")
        return expr

    def _ground_cardinality_constraints(self):
        """直接生成基数约束的CNF子句并添加到self.clauses中。"""
        for pred_map, op, bound in self.problem.cardinality_constraint.constraints:
            for pred, _ in pred_map.items():
                pred_name = str(pred)
                k = int(bound)

                related_vars = [v for ksym, v in self.sym_to_id.items() if pred_name in str(ksym)]
                if not related_vars:
                    continue

                # 使用pysat生成CNF子句
                if op == "<=":
                    cnf_cc = CardEnc.atmost(lits=related_vars, bound=k, top_id=self.next_var_id - 1)
                elif op == ">=":
                    cnf_cc = CardEnc.atleast(lits=related_vars, bound=k, top_id=self.next_var_id - 1)
                elif op == "=":
                    cnf_cc = CardEnc.equals(lits=related_vars, bound=k, top_id=self.next_var_id - 1)
                else:
                    raise RuntimeError(f"Unknown operator: {op}")

                # **关键修复**: 直接将生成的子句添加到self.clauses，并更新变量计数器
                self.clauses.extend(cnf_cc.clauses)
                self.next_var_id = cnf_cc.nv
        
    # ... (build_counting, _build_eq, _build_mod, is_single_layer methods can remain as they are) ...
    def is_single_layer(self, formula):
        return isinstance(formula.quantified_formula, QFFormula)

    def build_counting(self, cnt_formulas, expr):
        single_layer_formulas, double_layer_formulas = [], []
        for f in cnt_formulas:
            (single_layer_formulas if self.is_single_layer(f) else double_layer_formulas).append(f)

        for formula in single_layer_formulas:
            scope = formula.quantifier_scope
            expr &= self._build_eq(formula.quantified_formula, None, scope.quantified_var, None, scope.count_param) if scope.comparator == '=' else self._build_mod(formula.quantified_formula, None, scope.quantified_var, None, scope.count_param)

        for e1 in self.domain:
            for formula in double_layer_formulas:
                scope = formula.quantified_formula.quantifier_scope
                qf = formula.quantified_formula.quantified_formula
                free_vars = qf.vars() - {scope.quantified_var}
                var_x = next(iter(free_vars)) if free_vars else None
                expr &= self._build_eq(qf, var_x, scope.quantified_var, e1, scope.count_param) if scope.comparator == '=' else self._build_mod(qf, var_x, scope.quantified_var, e1, scope.count_param)
        return expr

    def _build_eq(self, inner_qf, var_x, var_y, e1, k):
        clause_expr = sympy.false
        for combo in combinations(self.domain, k):
            sub_conj = sympy.true
            domain_set, combo_set = set(self.domain), set(combo)
            for y in combo_set:
                subst = {var_y: y, **({var_x: e1} if var_x else {})}
                g = inner_qf.substitute(subst)
                for atom in g.atoms(): self._register_atom(atom)
                sub_conj &= g.expr
            for y in domain_set - combo_set:
                subst = {var_y: y, **({var_x: e1} if var_x else {})}
                g = inner_qf.substitute(subst)
                for atom in g.atoms(): self._register_atom(atom)
                sub_conj &= ~g.expr
            clause_expr |= sub_conj
        return sympy.simplify_logic(clause_expr, form='cnf')

    def _build_mod(self, inner_qf, var_x, var_y, e1, rk):
        r, k_mod = rk
        clause_expr = sympy.false
        for n in range(len(self.domain) + 1):
            if n % k_mod != r: continue
            for combo in combinations(self.domain, n):
                sub_conj = sympy.true
                domain_set, combo_set = set(self.domain), set(combo)
                for y in combo_set:
                    subst = {var_y: y, **({var_x: e1} if var_x else {})}
                    g = inner_qf.substitute(subst)
                    for atom in g.atoms(): self._register_atom(atom)
                    sub_conj &= g.expr
                for y in domain_set - combo_set:
                    subst = {var_y: y, **({var_x: e1} if var_x else {})}
                    g = inner_qf.substitute(subst)
                    for atom in g.atoms(): self._register_atom(atom)
                    sub_conj &= ~g.expr
                clause_expr |= sub_conj
        return sympy.simplify_logic(clause_expr, form='cnf')

    def dump(self):
        """将 self.clauses 中的所有子句写入CNF文件。"""
        num_vars = self.next_var_id 
        num_clauses = len(self.clauses)
        with open(self.cnf_path, 'w', encoding='utf-8') as f:
            f.write(f"p cnf {num_vars} {num_clauses}\n")
            for clause in self.clauses:
                f.write(" ".join(map(str, clause)) + " 0\n")
        
        logger.info(f"CNF file with {num_vars} vars and {num_clauses} clauses written to: {self.cnf_path}")

    @staticmethod
    def model_count_pysat(cnf_path: str) -> int:
        # ... (static methods are fine)
        cnf = CNF(from_file=cnf_path)
        count = 0
        with Solver(bootstrap_with=cnf) as s:
            for _ in s.enum_models():
                count += 1
        return count

    @staticmethod
    def model_count_ganak(cnf_path: str) -> int:
        # ... (static methods are fine)
        result = subprocess.run([ganak_path, cnf_path], capture_output=True, text=True)
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


def Fo2Counter(file_name: str, domain_size: int, counter: str = "pysat", epsilon=0.01, delta=0.01) -> int:
    context = CNFContext(file_name, domain_size)
    context.convert() # This single call now does everything.

    if counter == "pysat":
        return CNFContext.model_count_pysat(context.cnf_path)
    elif counter == "ganak":
        return CNFContext.model_count_ganak(context.cnf_path)
    elif counter == "approxmc":
        return context.model_count_approxmc(context.cnf_path, epsilon, delta)
    raise ValueError(f"Unknown counter: {counter}")

def main():
    log_path = os.path.join(CNF_RESULTS_PATH, "performance.log")
    logfile(log_path, maxBytes=1e6, backupCount=3)
    logger.setLevel(logging.INFO)

    # file_name = "2-regular-graph.wfomcs"
    file_name = "m-odd-degree-graph-cc.wfomcs"
    # file_name= "2-regular-graph-sc2.wfomcs"
    # file_name = "universal_quantifier_example.wfomcs"
    # file_name = "cardinality_constraints_example.wfomcs"
    domain_size = 3
    counters = ["ganak"]
    # counters = ["ganak", "pysat", "approxmc"]
    for counter in counters:
        try:
            count = Fo2Counter(file_name, domain_size, counter)
            print(f"File: {file_name}, Domain Size: {domain_size}, Counter: {counter}, Model Count: {count}")
        except Exception as e:
            logger.error(f"An error occurred for {counter}: {e}", exc_info=True)
            print(f"An error occurred for counter {counter}. Check logs.")

if __name__ == '__main__':
    main()