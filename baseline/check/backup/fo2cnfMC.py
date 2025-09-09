#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FO2到CNF转换及模型计数工具

文件路径：check/fo2_to_cnf_converter.py

功能概述：
本脚本实现了二阶逻辑(FO2)到合取范式(CNF)的转换，并支持模型计数功能，主要特点包括：
1. 支持带计数约束的量词（=k 和 r mod k）
2. 自动生成符合DIMACS标准的CNF文件
3. 集成PySAT模型计数功能
4. 提供完整的命令行接口

核心功能：
- 将FO2公式转换为等价的CNF形式
- 支持基数约束(∃=k)和同余约束(∃ r mod k)
- 自动处理变量替换和公式展开
- 生成兼容sharpSAT等求解器的CNF文件
- 内置PySAT模型计数器

使用方法：
1. 命令行调用：
   python fo2_to_cnf_converter.py -i <输入文件> -n <域大小>

2. 参数说明：
   -i/--input: 输入文件路径（FO2公式文件）
   -n/--domain-size: 论域大小（正整数）

3. 输出结果：
   - 生成CNF文件：check/原文件名_域大小.cnf
   - 控制台输出模型计数结果

工作流程：
1. 解析输入公式和域大小
2. 处理三种约束类型：
   - 基础公式(universal部分)
   - 存在量词扩展公式
   - 计数约束公式
3. 生成DIMACS格式CNF文件
4. 使用PySAT进行模型计数

注意事项：
1. 输入文件需符合特定语法格式
2. 大域大小可能导致组合爆炸
3. 生成的CNF文件可直接用于其他求解器
4. 计数约束支持全局计数和单变量计数
"""
import os
import copy
import argparse
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


###############################################################################
# 工具函数
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description='Convert a first-order logic sentence to CNF')
    parser.add_argument('--input', '-i', type=str, required=True, help='sentence file')
    parser.add_argument('--domain-size', '-n', type=int, required=True, help='domain size')
    return parser.parse_args()


def get_problem(input_path: str, domain_size: int) -> WFOMCProblem:
    problem = parse_input(input_path)
    domain = {Const(str(i)) for i in range(domain_size)}
    problem.domain = domain
    return problem


def _register_atoms(qf: QFFormula, atom_to_digit: Dict, atomsym_to_digit: Dict) -> None:
    """
    将 *新的* 原子加入 DIMACS 编号表。

    参数说明
    ----------
    qf : QFFormula
        量化自由子公式；其中的 atoms() 方法返回 AtomicFormula 集合。
    atom_to_digit / atomsym_to_digit : dict
        两张互补的映射表，分别按 *AtomicFormula* 与 *sympy.Symbol* 索引。
    """
    for atom in qf.atoms():
        if atom not in atom_to_digit:
            idx = len(atom_to_digit) + 1  # DIMACS 变量编号从 1 开始
            atom_to_digit[atom] = idx
            atomsym_to_digit[atom.expr] = idx


def _extract_qf(formula):
    """递归剥离最外层量词，直到获取量化自由公式。"""
    while not isinstance(formula, QFFormula):
        formula = formula.quantified_formula
    return formula


###############################################################################
# 分步骤处理函数
###############################################################################

def _ground_universal(uni_qf: QFFormula, domain: Set, expr, at2d, sym2d):
    """处理 ∀X∀Y 基础子句。"""
    for e1, e2 in product(domain, repeat=2):
        grounded = uni_qf.substitute({X: e1, Y: e2}) & \
                   uni_qf.substitute({X: e2, Y: e1})
        g_expr = grounded.expr

        # 处理 ⊤ / ⊥ 两种特殊情况
        if g_expr is None:
            if grounded is top:
                continue  # ⊤ 对合取无影响
            return sympy.false  # 出现 ⊥，整个公式恒假，直接回传

        expr &= sympy.to_cnf(g_expr)
        _register_atoms(grounded, at2d, sym2d)
    return expr


def _ground_extensions(problem: WFOMCProblem, domain: Set, expr, at2d, sym2d):
    """处理扩展公式（纯 ∃Y 形式）。"""
    ext_qfs: Iterable[QFFormula] = [
        _extract_qf(copy.deepcopy(f.quantified_formula))
        for f in problem.sentence.ext_formulas
    ]
    for e1 in domain:
        for ext_qf in ext_qfs:
            # ∃Y: φ(X,Y)  ---->  ⋁_{y∈Δ} φ(e1,y)
            disjunction = sympy.false
            for e2 in domain:
                grounded = ext_qf.substitute({X: e1, Y: e2})
                disjunction |= grounded.expr
                _register_atoms(grounded, at2d, sym2d)
            expr &= disjunction
    return expr


def _encode_counting(problem: WFOMCProblem, domain: Set, expr, at2d, sym2d):
    """编码计数量词 =k 与 r mod k。"""
    cnt_formulas = copy.deepcopy(problem.sentence.cnt_formulas)

    for e1 in domain:  # 外层 ∀X 构造
        for cnt_formula in cnt_formulas:
            q_scope = cnt_formula.quantified_formula.quantifier_scope
            var_y = q_scope.quantified_var  # 被计数变量 Y
            comparator = q_scope.comparator  # '=' 或 'mod'
            param = q_scope.count_param  # k  或  (r,k)
            inner_qf = cnt_formula.quantified_formula.quantified_formula

            # ---------------- 找到自由变量（X 或全局） ----------------
            free_vars = inner_qf.vars() - {var_y}
            var_x = next(iter(free_vars)) if free_vars else None

            # ---------------- 根据比较符生成约束 ----------------
            if comparator == '=':
                expr &= _build_eq_constraint(inner_qf, var_x, var_y, e1, param,
                                             domain, at2d, sym2d)
            elif comparator == 'mod':
                expr &= _build_mod_constraint(inner_qf, var_x, var_y, e1, param,
                                              domain, at2d, sym2d)
            else:
                raise NotImplementedError(f"Unsupported comparator '{comparator}'.")
    return expr


###############################################################################
# 计数子句生成器
###############################################################################

def _build_eq_constraint(inner_qf: QFFormula, var_x, var_y, e1, k_val: int,
                         domain: Set, at2d, sym2d):
    """生成 ∃_{=k} 子句。"""
    clause_expr = sympy.false
    for combo in combinations(domain, k_val):
        sub_conj = sympy.true
        # --- ① 选中的 Y 置真 ---
        for y in combo:
            subst = {var_y: y}
            if var_x is not None:
                subst[var_x] = e1
            g = inner_qf.substitute(subst)
            sub_conj &= g.expr
            _register_atoms(g, at2d, sym2d)
        # --- ② 未选中的 Y 置假 ---
        for y in set(domain) - set(combo):
            subst = {var_y: y}
            if var_x is not None:
                subst[var_x] = e1
            g = inner_qf.substitute(subst)
            sub_conj &= sympy.Not(g.expr)
            _register_atoms(g, at2d, sym2d)
        clause_expr |= sub_conj
    return sympy.simplify_logic(clause_expr, form='cnf')


def _build_mod_constraint(inner_qf: QFFormula, var_x, var_y, e1,
                          rk_pair: Tuple[int, int], domain: Set, at2d, sym2d):
    """生成 ∃_{r mod k} 子句。"""
    r, k_mod = rk_pair
    clause_expr = sympy.false
    for n in range(len(domain) + 1):
        if n % k_mod != r:
            continue
        for combo in combinations(domain, n):
            sub_conj = sympy.true
            for y in combo:
                subst = {var_y: y}
                if var_x is not None:
                    subst[var_x] = e1
                g = inner_qf.substitute(subst)
                sub_conj &= g.expr
                _register_atoms(g, at2d, sym2d)
            for y in set(domain) - set(combo):
                subst = {var_y: y}
                if var_x is not None:
                    subst[var_x] = e1
                g = inner_qf.substitute(subst)
                sub_conj &= sympy.Not(g.expr)
                _register_atoms(g, at2d, sym2d)
            clause_expr |= sub_conj
    return sympy.simplify_logic(clause_expr, form='cnf')


###############################################################################
# 输出 DIMACS 文件
###############################################################################

def _dump_dimacs(st, out_path: str):
    """将 st['expr'] 写成 DIMACS。自动过滤空子句。"""
    expr_cnf = sympy.to_cnf(st['expr'])

    # -- collect clause nodes -------------------------------------------------
    if isinstance(expr_cnf, sympy.And):
        raw_clauses = list(expr_cnf.args)
    else:
        raw_clauses = [expr_cnf]

    dimacs_lines: List[str] = []
    empty_cnt = 0

    for cl in raw_clauses:
        lits: List[str] = []
        atoms = cl.args if isinstance(cl, sympy.Or) else [cl]
        for at in atoms:
            if isinstance(at, sympy.Symbol):
                lit_id = st['sym2id'].get(at, None)
                if lit_id is None:
                    # should never happen
                    raise KeyError(f"Unmapped atom {at}")
                lits.append(str(lit_id))
            elif isinstance(at, sympy.Not):
                base = ~at  # 取除去最外层 Not 的 Symbol
                lit_id = st['sym2id'].get(base, None)
                if lit_id is None:
                    raise KeyError(f"Unmapped ¬‑atom {base}")
                lits.append(str(-lit_id))
            else:
                # SymPy 可能把 (p | ¬p) 化简掉；若出现 True / False，继续
                if at == sympy.true:
                    # (l ∨ ¬l) ≡ ⊤ —— 整句为真，可跳过
                    lits = ['t']  # 标记以便后续丢弃整子句
                    break
                elif at == sympy.false:
                    # 单独 False ⇒ 空子句 → 计数器
                    continue
                else:
                    raise RuntimeError(f"未知原子类型: {type(at)} → {at}")

        # 处理收集结果
        if 't' in lits:
            continue  # tautology, 跳过
        if not lits:
            empty_cnt += 1  # 记录空子句
            continue
        dimacs_lines.append(' '.join(lits) + ' 0')

    # ---------- UNSAT 检测 ----------
    if empty_cnt > 0 and not dimacs_lines:
        # 整体不可满足；输出一个注释文件即可
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('c unsat\n')
            f.write('p cnf 1 1\n0\n')
        logger.warning('[UNSAT] 全公式含 %d 个空子句，已写入占位文件 %s', empty_cnt, out_path)
        return False  # 表示未生成可解 CNF

    # ---------- 正常写文件 ----------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"c atoms: {' '.join(map(str, st['atom2id'].keys()))}\n")
        f.write(f"p cnf {len(st['atom2id'])} {len(dimacs_lines)}\n")
        f.write('\n'.join(dimacs_lines) + '\n')

    if empty_cnt:
        logger.warning('Detected %d empty clause(s); 已过滤', empty_cnt)
    else:
        logger.info('DIMACS written → %s', out_path)
    return True


###############################################################################
# 主入口 —— 供外部调用
###############################################################################

def convert_to_cnf(problem: WFOMCProblem, output_path: str):
    """将 WFOMCProblem 转换为 DIMACS CNF 文件。"""
    domain = problem.domain

    # ===== 公共状态初始化 =====
    expr = sympy.true  # 全局合取公式
    atom_to_digit: Dict = {}  # AtomicFormula -> int
    atomsym_to_digit: Dict = {}  # sympy.Symbol  -> int

    # ===== 1) 基础 ∀∀ 子句 =====
    uni_qf = _extract_qf(copy.deepcopy(problem.sentence.uni_formula))
    expr = _ground_universal(uni_qf, domain, expr, atom_to_digit, atomsym_to_digit)
    if expr is sympy.false:
        _dump_dimacs(expr, output_path, atom_to_digit, atomsym_to_digit)
        return

    # ===== 2) 扩展公式 =====
    expr = _ground_extensions(problem, domain, expr, atom_to_digit, atomsym_to_digit)

    # ===== 3) 计数量词 =====
    expr = _encode_counting(problem, domain, expr, atom_to_digit, atomsym_to_digit)

    # ===== 4) 输出文件 =====
    _dump_dimacs(expr, output_path, atom_to_digit, atomsym_to_digit)

# 5. SAT counting helper
def count_with_pysat(path):
    cnf = CNF(from_file=path)
    count = 0
    with Solver(bootstrap_with=cnf) as s:
        for model in s.enum_models():
            count += 1
    return count


# ###############################################################################
if __name__ == "__main__":
    args = parse_args()
    # 获取问题实例：解析输入文件并设置域大小
    problem = get_problem(args.input, args.domain_size)
    # 提取输入文件的基本名称（不含路径）
    sentence_base = os.path.basename(args.input)
    # 构造输出CNF文件路径：check/原文件名_域大小.cnf
    output_file = os.path.join("check", f"{sentence_base[:-7]}_{args.domain_size}.cnf")
    # 检查CNF文件是否已存在，避免重复生成
    if not os.path.exists(output_file):
        # 调用核心转换函数生成CNF文件
        convert_to_cnf(problem, output_file)
    # 使用PySAT计算生成的CNF文件的模型数量
    mc = count_with_pysat(output_file)
    print(f"Models: {mc}")
