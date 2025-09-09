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

DIR_PATH = Path(__file__).parent

import re
import os, copy, sympy
from itertools import product, combinations
from pysat.formula import CNF
from pysat.solvers import Solver
from wfomc import parse_input
from wfomc.problems import WFOMCProblem
from wfomc.fol.syntax import Const, X, Y, QFFormula, top
import subprocess
from logzero import logger, logfile
import logging

ganak_path = "/home/sunshixin/software/ganak/ganak"  # 如果ganak在PATH中
approxmc_path = "/home/sunshixin/software/approxmc/approxmc"

class AtomManager:
    def __init__(self):
        self.atom_to_id = {}  # 初始化原子到ID的映射字典，用于将逻辑原子映射到DIMACS变量编号
        self.sym_to_id = {}  # 初始化符号到ID的映射字典，用于将sympy符号映射到DIMACS变量编号

    def register_atoms(self, qf: QFFormula):
        for atom in qf.atoms():  # 遍历量化自由公式中的所有原子
            if atom not in self.atom_to_id:  # 检查原子是否尚未注册到映射中
                idx = len(self.atom_to_id) + 1  # 为新原子分配一个新的ID，DIMACS变量编号从1开始
                self.atom_to_id[atom] = idx  # 将原子与其ID添加到原子到ID的映射字典中
                self.sym_to_id[atom.expr] = idx  # 将原子的表达式（sympy符号）与其ID添加到符号到ID的映射字典中


class CNFWriter:
    def __init__(self, atom_mgr: AtomManager):
        self.atom_mgr = atom_mgr  # 初始化CNF写入器，接收AtomManager实例用于获取原子到ID的映射

    def dump(self, expr, out_path: str):
        expr_cnf = sympy.to_cnf(expr)  # 将表达式转换为合取范式(CNF)形式
        clauses = expr_cnf.args if isinstance(expr_cnf, sympy.And) else [expr_cnf]  # 提取CNF中的所有子句，如果表达式是AND组合则获取其参数，否则作为一个单独的子句
        lines = []  # 初始化存储DIMACS格式行的列表
        for cl in clauses:  # 遍历每个子句
            atoms = cl.args if isinstance(cl, sympy.Or) else [cl]  # 提取子句中的所有原子，如果子句是OR组合则获取其参数，否则作为一个单独的原子
            lits = []  # 初始化存储文字(变量)的列表
            for at in atoms:  # 遍历每个原子
                if isinstance(at, sympy.Symbol):  # 如果原子是符号(变量)
                    lits.append(str(self.atom_mgr.sym_to_id[at]))  # 获取该符号对应的ID并添加到文字列表中
                elif isinstance(at, sympy.Not):  # 如果原子是否定形式
                    base = ~at  # 获取否定符号的基础符号
                    lits.append(str(-self.atom_mgr.sym_to_id[base]))  # 获取基础符号对应的ID，添加负号表示否定，并添加到文字列表中
            if lits: lines.append(" ".join(lits) + " 0")  # 如果文字列表非空，则将其格式化为DIMACS行格式并添加到行列表中
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"p cnf {len(self.atom_mgr.atom_to_id)} {len(lines)}\n")  # 写入DIMACS格式的头部信息，包括变量数量和子句数量
            f.write("\n".join(lines))
        logger.info(f"CNF written: {out_path}")  # 记录日志信息，表示CNF文件已写入


class Grounder:
    def __init__(self, atom_mgr: AtomManager, domain):
        self.atom_mgr = atom_mgr  # 初始化Grounder，接收AtomManager实例和域集合
        self.domain = domain

    def extract_qf(self, formula):  # 递归提取量化自由公式(QFFormula)，剥除所有量词直到得到核心公式
        while not isinstance(formula, QFFormula):
            formula = formula.quantified_formula
        return formula

    def ground_universal(self, qf: QFFormula, expr):
        for e1, e2 in product(self.domain, repeat=2):  # 处理全称量词公式，对域中所有元素组合进行展开
            grounded = qf.substitute({X: e1, Y: e2}) & qf.substitute({X: e2, Y: e1})  # 对每对域元素进行替换，并考虑对称性(X/Y交换)
            if grounded.expr is None:  # 检查展开后的公式是否为特殊值(None, top, false)
                if grounded is top: continue  # 如果是top(真)，则对合取无影响，继续下一个
                return sympy.false  # 如果是false(假)，整个公式恒假，直接返回
            expr &= sympy.to_cnf(grounded.expr)  # 将展开后的表达式转换为CNF形式并与总表达式合取
            self.atom_mgr.register_atoms(grounded)  # 注册新产生的原子到AtomManager中
        return expr

    def ground_extensions(self, ext_formulas, expr):  # 处理扩展公式(存在量词公式)
        ext_qfs = [self.extract_qf(copy.deepcopy(f.quantified_formula)) for f in ext_formulas]  # 提取所有扩展公式的量化自由部分
        for e1 in self.domain:  # 对域中的每个元素e1进行处理
            for ext_qf in ext_qfs:  # 遍历所有扩展公式
                disjunction = sympy.false  # 初始化析取表达式为false
                for e2 in self.domain:  # 对域中的每个元素e2进行存在量词展开
                    grounded = ext_qf.substitute({X: e1, Y: e2})  # 将Y替换为e2，形成一个具体实例
                    disjunction |= grounded.expr  # 将该实例的表达式加入析取
                    self.atom_mgr.register_atoms(grounded)  # 注册新产生的原子
                expr &= disjunction  # 将析取表达式与总表达式合取
        return expr

    def build_counting(self, cnt_formulas, expr):  # 构建计数约束公式
        for e1 in self.domain:  # 对域中的每个元素e1进行处理
            for cnt_formula in cnt_formulas:  # 遍历所有计数公式
                q_scope = cnt_formula.quantified_formula.quantifier_scope  # 提取量词作用域信息
                var_y = q_scope.quantified_var  # 获取被计数的变量Y
                comparator = q_scope.comparator  # 获取比较符(= 或 mod)
                param = q_scope.count_param  # 获取计数参数(k 或 (r,k))
                inner_qf = cnt_formula.quantified_formula.quantified_formula  # 获取量词内部的量化自由公式
                free_vars = inner_qf.vars() - {var_y}  # 确定自由变量(除了被计数变量Y之外的变量)
                var_x = next(iter(free_vars)) if free_vars else None  # 如果存在自由变量则获取，否则设为None
                if comparator == '=':  # 构建等于k的约束
                    expr &= self._build_eq(inner_qf, var_x, var_y, e1, param)
                elif comparator == 'mod':  # 构建模k余r的约束
                    expr &= self._build_mod(inner_qf, var_x, var_y, e1, param)
        return expr

    def _build_eq(self, inner_qf, var_x, var_y, e1, k):  # 构建等于k的计数约束
        clause_expr = sympy.false  # 初始化子句表达式为false
        for combo in combinations(self.domain, k):  # 遍历域中元素的所有k组合
            sub_conj = sympy.true  # 初始化子合取表达式为true
            for y in combo:  # 对于选中的元素(在combo中)，将其对应的原子设为真
                subst = {var_y: y, **({var_x: e1} if var_x else {})}  # 构造替换字典，包含Y和可能的X变量
                g = inner_qf.substitute(subst)  # 进行变量替换
                sub_conj &= g.expr  # 将原子表达式与子合取表达式合取
                self.atom_mgr.register_atoms(g)  # 注册新产生的原子
            for y in set(self.domain) - set(combo):  # 对于未选中的元素(不在combo中)，将其对应的原子设为假
                subst = {var_y: y, **({var_x: e1} if var_x else {})}  # 构造替换字典
                g = inner_qf.substitute(subst)  # 进行变量替换
                sub_conj &= sympy.Not(g.expr)  # 将原子的否定表达式与子合取表达式合取
                self.atom_mgr.register_atoms(g)  # 注册新产生的原子
            clause_expr |= sub_conj  # 将子合取表达式与子句表达式析取
        return sympy.simplify_logic(clause_expr, form='cnf')  # 简化逻辑表达式并转换为CNF形式

    def _build_mod(self, inner_qf, var_x, var_y, e1, rk):  # 构建模k余r的计数约束
        r, k_mod = rk  # 解析参数(r, k)
        clause_expr = sympy.false  # 初始化子句表达式为false
        for n in range(len(self.domain) + 1):  # 遍历可能的计数值n(从0到域大小)
            if n % k_mod != r: continue  # 检查n是否满足模约束条件，不满足则跳过
            for combo in combinations(self.domain, n):  # 遍历域中元素的所有n组合
                sub_conj = sympy.true  # 初始化子合取表达式为true
                for y in combo:  # 对于选中的元素(在combo中)，将其对应的原子设为真
                    subst = {var_y: y, **({var_x: e1} if var_x else {})}  # 构造替换字典
                    g = inner_qf.substitute(subst)  # 进行变量替换
                    sub_conj &= g.expr  # 将原子表达式与子合取表达式合取
                    self.atom_mgr.register_atoms(g)  # 注册新产生的原子
                for y in set(self.domain) - set(combo):  # 对于未选中的元素(不在combo中)，将其对应的原子设为假
                    subst = {var_y: y, **({var_x: e1} if var_x else {})}  # 构造替换字典
                    g = inner_qf.substitute(subst)  # 进行变量替换
                    sub_conj &= sympy.Not(g.expr)  # 将原子的否定表达式与子合取表达式合取
                    self.atom_mgr.register_atoms(g)  # 注册新产生的原子
                clause_expr |= sub_conj  # 将子合取表达式与子句表达式析取
        return sympy.simplify_logic(clause_expr, form='cnf')  # 简化逻辑表达式并转换为CNF形式


class FO2CNFConverter:
    def __init__(self, file_name, domain_size):
        input_path = os.path.join(DIR_PATH, "models", file_name)  # 构造输入文件的完整路径，从models目录中读取
        self.problem = parse_input(input_path)  # 解析输入文件，获取WFOMC问题实例
        self.problem.domain = {Const(str(i)) for i in range(domain_size)}
        self.atom_mgr = AtomManager()  # 初始化原子管理器，用于管理逻辑原子与DIMACS变量ID的映射
        self.grounder = Grounder(self.atom_mgr, self.problem.domain)  # 初始化公式展开器，用于将量化公式展开为具体的逻辑表达式
        self.writer = CNFWriter(self.atom_mgr)  # 初始化CNF写入器，用于将逻辑表达式写入CNF文件

    def convert(self, out_name):
        out_path = os.path.join(DIR_PATH, "cnf_results", out_name)  # 构造完整的输出路径
        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        expr = sympy.true  # 初始化表达式为真(true)，作为所有子句的合取起点
        uni_qf = self.grounder.extract_qf(copy.deepcopy(self.problem.sentence.uni_formula))  # 提取全称量词公式的量化自由部分
        expr = self.grounder.ground_universal(uni_qf, expr)  # 展开全称量词公式并将其合取到总表达式中
        expr = self.grounder.ground_extensions(self.problem.sentence.ext_formulas, expr)  # 展开存在量词(扩展)公式并将其合取到总表达式中
        expr = self.grounder.build_counting(self.problem.sentence.cnt_formulas, expr)  # 构建计数约束公式并将其合取到总表达式中
        self.writer.dump(expr, out_path)  # 将最终的逻辑表达式写入CNF文件


class CNFCounter:
    @staticmethod
    def model_count_pysat(out_name):
        # 构造完整的输出路径
        cnf_path = os.path.join(DIR_PATH, "results", out_name)

        cnf = CNF(from_file=cnf_path)
        count = 0
        with Solver(bootstrap_with=cnf) as s:
            for _ in s.enum_models():
                count += 1
        return count

    @staticmethod
    def model_count_ganak(file_name: str) -> int:
        """
        使用Ganak求解器进行模型计数
        """
        # 构造完整的输出路径
        cnf_path = os.path.join(DIR_PATH, "cnf_results", file_name)

        result = subprocess.run([ganak_path, cnf_path], stdout=subprocess.PIPE, text=True)

        if result.returncode == 0:
            # 解析Ganak的输出以提取模型计数
            # Ganak的输出格式通常包含类似 "s mc <count>" 的行
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if line.startswith('s mc'):
                    # 提取模型计数
                    count = int(line.split()[2])
                    return count
                elif 'exact arb int' in line:
                    try:
                        # 使用正则表达式匹配 "c s exact arb int <数字>"
                        match = re.search(r'c s exact arb int (\d+)', line)
                        if match:
                            count = int(match.group(1))
                            return count
                    except (IndexError, ValueError) as e:
                        logger.warning(f"解析'c s exact arb int'行失败: {line}, 错误: {e}")

            # 如果没有找到标准格式，尝试其他可能的输出格式
            # Ganak有时会输出类似 "# solutions = <count>" 的行
            for line in output_lines:
                if "solutions" in line and "=" in line:
                    count = int(line.split('=')[1].strip())
                    return count

            logger.warning("无法从Ganak输出中解析模型计数")
            return 0
        else:
            logger.error(f"Ganak执行失败: {result.stderr}")
            return 0

    @staticmethod
    def model_count_approxmc(cnf_file_name: str, epsilon, delta) -> dict:
        """
        使用ApproxMC进行近似模型计数
        """
        # 构造完整的CNF文件路径
        cnf_path = os.path.join(DIR_PATH, "cnf_results", cnf_file_name)

        cmd = [
            approxmc_path,
            f"--epsilon={epsilon}",
            f"--delta={delta}",
            cnf_path
        ]

        # result = subprocess.run([approxmc_path, cnf_path], stdout=subprocess.PIPE, text=True)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            if line.startswith("s mc"):
                return int(line.split()[-1])  # 提取最后的数字
        raise RuntimeError("ApproxMC 未返回计数结果")


# 主函数
def fo2_to_cnf_count(file_name: str, domain_size: int, counter: str = "pysat", epsilon=0.01, delta=0.01) -> int:
    """将FO2文件转换为CNF文件，并使用指定的计数器进行模型计数"""
    converter = FO2CNFConverter(file_name, domain_size)
    cnf_file_name = os.path.splitext(file_name)[0]+f"_domain_size_{domain_size}" + ".cnf"
    out_path = os.path.join(DIR_PATH, "cnf_results", cnf_file_name)  # 构造完整的输出路径
    if not os.path.exists(out_path):  # 判断result中是否已经存在CNF文件
        converter.convert(cnf_file_name)
    if counter == "pysat":
        return CNFCounter.model_count_pysat(cnf_file_name)  # 默认使用PySAT
    elif counter == "ganak":
        return CNFCounter.model_count_ganak(cnf_file_name)
    elif counter == "approxmc":
        return CNFCounter.model_count_approxmc(cnf_file_name, epsilon, delta)


class ExperimentConfig:
    """实验配置类"""
    # 路径配置
    global DIR_PATH
    MODELS_PATH = os.path.join(DIR_PATH, "models")  # 输入例子的路径
    CNF_RESULTS_PATH = os.path.join(DIR_PATH, "cnf_results")  # 存储实验结果的路径


if __name__ == '__main__':
    log_path = os.path.join(ExperimentConfig.RESULTS_PATH, "performance.log")
    # 使用disableStderrLogger=True参数来禁用控制台输出，只写入文件

    logfile(log_path, maxBytes=1e6, backupCount=3)
    # 设置日志等级（只显示 INFO 及以上）
    logger.setLevel(logging.INFO)

    file_name = "2-regular-graph.wfomcs"
    domain_size = 3  # 设置论域大小



    # 使用不同的 counter 获取 count 数值
    counters = ["pysat", "ganak", "approxmc"]
    for counter in counters:
        count = fo2_to_cnf_count(file_name, domain_size, counter)
        print(f"Model count using {counter}: {count}")
