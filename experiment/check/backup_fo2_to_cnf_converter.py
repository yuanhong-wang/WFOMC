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
- 支持基数约束(∃=k)和mod约束(∃ r mod k)
- 自动处理变量替换和公式展开
- 生成兼容sharpSAT等求解器的CNF文件
- 内置PySAT模型计数器



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
from itertools import product, combinations, combinations_with_replacement
from pysat.card import CardEnc, EncType


MODEL_DIR_PATH = "/home/sunshixin/pycharm_workspace/WFOMC/models" # 这是输入文件的路径，需要根据实际路径修改
DIR_PATH = Path(__file__).parent # 这是保存中间结果CNF输出结果的路径，然后后续进行操作
ganak_path = "/home/sunshixin/software/ganak/ganak"  # 如果ganak在linux服务器的PATH中
approxmc_path = "/home/sunshixin/software/approxmc/approxmc" # 如果approxmc在PATH中

class AtomManager:
    def __init__(self):
        self.atom_to_id = {}  # 初始化原子到ID的映射字典，用于将逻辑原子映射到DIMACS变量编号
        self.sym_to_id = {}  # 初始化符号到ID的映射字典，用于将sympy符号映射到DIMACS变量编号

    def register_atoms(self, qf: QFFormula):
        for atom in qf.atoms():  # 遍历无全称量词公式中的所有原子
            if atom not in self.atom_to_id:  # 检查原子是否尚未注册到映射中
                idx = len(self.atom_to_id) + 1  # 为新原子分配一个新的ID，DIMACS变量编号从1开始
                self.atom_to_id[atom] = idx  # 将原子与其ID添加到原子到ID的映射字典中
                self.sym_to_id[atom.expr] = idx  # 将原子的表达式（sympy符号）与其ID添加到符号到ID的映射字典中


class CnfFileWriter:
    """
    CNF写入器类：负责将逻辑表达式转换为DIMACS格式的CNF文件
    该类使用AtomManager来获取逻辑原子到DIMACS变量ID的映射
    """
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


class Context:
    """
    公式展开器类：负责将FO2公式展开为具体的逻辑表达式
    """
    def __init__(self, atom_mgr: AtomManager, domain):
        self.atom_mgr = atom_mgr  # 初始化Context，接收AtomManager实例和域集合
        self.domain = domain

    def extract_qf(self, formula):  # 递归提取量无量词公式(QFFormula)，剥除所有量词直到得到核心公式
        while not isinstance(formula, QFFormula):
            formula = formula.quantified_formula
        return formula

    def ground_universal(self, qf: QFFormula, expr):
        for e1, e2 in combinations_with_replacement(self.domain, 2):  # 处理全称量词公式，对域中所有元素组合进行展开。domain中元素用数字表示。
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
    
    def ground_cardinality_constraints(self, problem, expr):
        if problem.cardinality_constraint is None:
            return expr
        # 处理基数约束
        for cc in problem.cardinality_constraint.constraints:
            expr &= self._build_cardinality_constraint(cc[0])
        return expr
    
    def _build_cardinality_constraint(self, constraints):
        cc_clauses = [] # 初始化一个列表，用于存放从基数约束转换来的所有CNF子句。
        for pred_map, op, bound in constraints: # 遍历每一个约束。每个约束是一个元组，如 ({Pred:"P"}, "<=", 5)。
            for pred, coeff in pred_map.items(): # 遍历约束中涉及的谓词。这段代码的结构暗示每个约束只处理一个谓词。
                pred_name = str(pred) # 获取谓词的名称，例如 "P"。
                k = int(bound)  # 获取约束的数值界限，例如 5。

                
                vars = [v for ksym, v in self.atom_mgr.sym_to_id.items() if pred_name in str(ksym)] # 这一行非常关键：它从已经建立的变量映射中，找出所有与当前谓词相关的变量。# 例如，如果谓词是 "P"，它会找到 P(c1), P(c2), ... 等所有基化原子对应的整数变量。
                if not vars: # 如果这个谓词没有任何基化原子（即没有对应的变量），则跳过。
                    continue
                if op == "<=": # 根据操作符（"<= ", ">=", "="）调用pysat库的CardEnc来生成CNF子句。
                    cnf_cc = CardEnc.atmost(lits=vars, bound=k, encoding=EncType.seqcounter    ) # 生成“最多k个”约束的CNF子句。
                elif op == ">=":
                    cnf_cc = CardEnc.atleast(lits=vars, bound=k, encoding=EncType.seqcounter    ) # 生成“最少k个”约束的CNF子句。
                elif op == "=":
                    cnf_cc = CardEnc.equals(lits=vars, bound=k, encoding=EncType.seqcounter    ) # 生成“正好k个”约束的CNF子句。
                else:
                    raise RuntimeError(f"Unknown operator: {op}") # 如果遇到未知的操作符，则抛出错误。
                
                modify_clauses=cnf_cc.clauses # 获取CardEnc生成的子句列表。这些子句可能包含原始变量和一些辅助变量。
                ignore_atom=vars # 创建一个列表，用于记录哪些变量是已知的（原始变量），哪些是新引入的辅助变量。
                
                #print('before:',len(atom_to_digit), modify_clauses)
                for clauses in modify_clauses: # 遍历生成的所有子句。
                    for i in clauses: # 遍历子句中的每个文字（变量或其否定）。
                        # 检查这个文字对应的变量是否是一个新引入的辅助变量。
                        # abs(i)是变量的整数ID。如果它不在已知变量列表ignore_atom中，说明是新的。
                        if abs(i) not in ignore_atom:
                            # 为这个新的辅助变量创建一个虚拟的原子公式，以便将其注册到全局变量映射中。
                            ccatom:AtomicFormula= AtomicFormula( Pred('CC'+str(len(self.atom_mgr.atom_to_id)+1), 1), 'c',True)
                            self.atom_mgr.atom_to_id[ccatom] = len(self.atom_mgr.atom_to_id)+1 # 将新原子和它的新整数ID添加到全局映射中。
                            self.atom_mgr.sym_to_id[ccatom.expr] = len(self.atom_mgr.sym_to_id)+1
                            ignore_atom.append(len(self.atom_mgr.atom_to_id)) # 将这个新的辅助变量ID添加到ignore_atom列表中，避免重复处理。

                            for j in modify_clauses: # 再次遍历所有子句，将pysat临时分配的辅助变量ID替换为我们刚刚注册的全局唯一ID。
                                for k_idx, k in enumerate(j): 
                                    if abs(i) == abs(k):
                                        j[k_idx] = len(self.atom_mgr.atom_to_id) if k > 0 else -len(self.atom_mgr.atom_to_id)

                #print('after:',len(atom_to_digit), modify_clauses)

                cc_clauses.extend(modify_clauses) # 将处理完（即所有变量ID都已全局注册）的子句添加到cc_clauses列表中。


    def is_single_layer(self, formula):
        """
        判断一个计数公式是否是单层（全局）的。
        单层: exists_{=k} X: P(X) -> formula 对象本身有 quantifier_scope
        双层: forall X: (exists_{=k} Y: B(X,Y)) -> formula 对象没有 quantifier_scope
        """
        return isinstance(formula.quantified_formula, QFFormula)

    def build_counting(self, cnt_formulas, expr):  # 构建计数约束公式
        single_layer_formulas = []
        double_layer_formulas = []

        # 1. 将计数公式分类
        for f in cnt_formulas:
            if self.is_single_layer(f):
                single_layer_formulas.append(f)
            else:
                double_layer_formulas.append(f)

        # 2. 处理单层（全局）约束，这部分不需要外层循环
        for formula in single_layer_formulas:
            var_y = formula.quantifier_scope.quantified_var
            comparator = formula.quantifier_scope.comparator
            param = formula.quantifier_scope.count_param
            inner_qf = formula.quantified_formula
            
            # 全局约束没有自由变量 var_x 和外部实例 e1
            var_x = None
            e1 = None

            if comparator == '=':
                expr &= self._build_eq(inner_qf, var_x, var_y, e1, param)
            elif comparator == 'mod':
                expr &= self._build_mod(inner_qf, var_x, var_y, e1, param)

        # 3. 处理双层（嵌套）约束，这部分需要外层循环
        for e1 in self.domain:
            for formula in double_layer_formulas:
                inner_q_scope = formula.quantified_formula.quantifier_scope
                var_y = inner_q_scope.quantified_var
                comparator = inner_q_scope.comparator
                param = inner_q_scope.count_param
                inner_qf = formula.quantified_formula.quantified_formula
                
                # 确定自由变量 var_x
                free_vars = inner_qf.vars() - {var_y}
                var_x = next(iter(free_vars)) if free_vars else None

                if comparator == '=':
                    expr &= self._build_eq(inner_qf, var_x, var_y, e1, param)
                elif comparator == 'mod':
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
    """
    FO2到CNF转换器类：负责将FO2公式转换为等价的CNF形式，并生成DIMACS
    """
    def __init__(self, file_name, domain_size):
        input_path = os.path.join(MODEL_DIR_PATH, file_name)  # 构造输入文件的完整路径，从models目录中读取
        self.problem = parse_input(input_path)  # 解析输入文件，获取WFOMC问题实例
        self.problem.domain = {Const(str(i)) for i in range(domain_size)} # 设置论域为指定大小的常量集合
        self.atom_mgr = AtomManager()  # 初始化原子管理器，用于管理逻辑原子与DIMACS变量ID的映射
        self.grounder = Context(self.atom_mgr, self.problem.domain)  # 初始化公式展开器，用于将量化公式展开为具体的逻辑表达式
        self.writer = CnfFileWriter(self.atom_mgr)  # 初始化CNF写入器，用于将逻辑表达式写入CNF文件

    def convert(self, out_name):
        out_path = os.path.join(DIR_PATH, "cnf_results", out_name)  # 构造中间结果CNF文件的完整的输出路径
        os.makedirs(os.path.dirname(out_path), exist_ok=True) # 确保输出目录存在
        # 开始转换，也就是根据problem中的uni_formula和ext_formulas和cnt_formulas分别进行处理。
        expr = sympy.true  # 初始化表达式为真(true)，作为所有子句的合取起点
        uni_qf = self.grounder.extract_qf(copy.deepcopy(self.problem.sentence.uni_formula))  # 提取全称量词公式的无量词部分 \forall X: \forall Y: P(X,Y) -> P(X,Y)
        expr = self.grounder.ground_universal(uni_qf, expr)  # 展开全称量词公式并将其合取到总表达式中
        expr = self.grounder.ground_extensions(self.problem.sentence.ext_formulas, expr)  # 展开存在量词(扩展)公式并将其合取到总表达式中
        expr = self.grounder.build_counting(self.problem.sentence.cnt_formulas, expr)  # 构建计数约束公式并将其合取到总表达式中
        expr = self.grounder.ground_cardinality_constraints(self.problem, expr)  # 处理基数约束公式（如果有的话）
        self.writer.dump(expr, out_path)  # 将最终的逻辑表达式写入CNF文件



# 主函数
def Fo2Counter(file_name: str, domain_size: int, counter: str = "pysat", epsilon=0.01, delta=0.01) -> int:
    """将FO2文件转换为CNF文件，并使用指定的计数器进行模型计数"""
    converter = FO2CNFConverter(file_name, domain_size) # 首先初始化转换器 的一些参数
    cnf_file_name = os.path.splitext(file_name)[0]+f"_domain_size_{domain_size}" + ".cnf" # 构造CNF文件名
    out_path = os.path.join(DIR_PATH, "cnf_results", cnf_file_name)  # 构造CNF完整的输出路径
    
    ## 1 先转换FO2到CNF
    # if not os.path.exists(out_path):  # 判断result中是否已经存在CNF文件
    #     converter.convert(cnf_file_name) # 这里直接调用上面的转换器进行转换
    converter.convert(cnf_file_name)
    
    ## 2 然后使用指定的计数器进行模型计数
    if counter == "pysat":
        return CnfFileCounter.model_count_pysat(cnf_file_name)  # 默认使用PySAT
    elif counter == "ganak":
        return CnfFileCounter.model_count_ganak(cnf_file_name) # 由于Ganak和Approxmc 各自的API的输出也是不同的，所以要用一个函数提取输出的结果
    elif counter == "approxmc":
        return CnfFileCounter.model_count_approxmc(cnf_file_name, epsilon, delta)

class CnfFileCounter:
    """
    这里是从CNF文件进行model counting的类
    """
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            if line.startswith("s mc"):
                return int(line.split()[-1])  # 提取最后的数字
        raise RuntimeError("ApproxMC 未返回计数结果")


class ExperimentConfig:
    """实验配置类"""
    # 路径配置
    global DIR_PATH
    MODELS_PATH = os.path.join(MODEL_DIR_PATH, "models")  # 输入例子的路径
    CNF_RESULTS_PATH = os.path.join(DIR_PATH, "cnf_results")  # 存储实验结果的路径
    RESULTS_PATH = CNF_RESULTS_PATH

def main():
    log_path = os.path.join(ExperimentConfig.RESULTS_PATH, "performance.log")
    # 使用disableStderrLogger=True参数来禁用控制台输出，只写入文件
    logfile(log_path, maxBytes=1e6, backupCount=3)
    logger.setLevel(logging.INFO) # 设置日志等级（只显示 INFO 及以上）

    # file_name = "m-odd-degree-graph-sc2.wfomcs"
    # file_name = "0mod2-regular-graph-sc2.wfomcs"
    file_name = "2-regular-graph.wfomcs"
    # file_name = "universal_quantifier_example.wfomcs"
    domain_size = 5  # 设置论域大小
    
    # 使用不同的 counter 获取 count 数值
    # counters = ["pysat", "ganak", "approxmc"]
    counters = ["ganak"]
    for counter in counters:
        count = Fo2Counter(file_name, domain_size, counter) # 调用核心函数
        print(f"Model count using {counter}: {count}") # 打印结果


if __name__ == '__main__':
    main()
