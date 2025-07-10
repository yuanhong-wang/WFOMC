"""
用途：将FO2 转为CNF，支持计数量词 =k 和 r mod k
使用：命令行调用：-i 文件名称 -n 域大小
     生成的文件在 check 目录下，文件名为：原文件名_域大小.cnf
"""
import os
import copy
import argparse
import sympy
from itertools import product, combinations
from logzero import logger, loglevel
from wfomc import parse_input
from wfomc.problems import WFOMCProblem
from wfomc.fol.syntax import Const, X, Y, QFFormula, AtomicFormula
import subprocess
from itertools import combinations
import subprocess
from logzero import logger
from pysat.formula import CNF
from pysat.solvers import Solver
import pandas as pd


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


def extract_qf_formula(formula):
    """Strip *all* leading quantifiers until we reach a `QFFormula`."""
    while not isinstance(formula, QFFormula):
        formula = formula.quantified_formula
    return formula


# ----------------------------------------------------------------------
# MAIN WORKHORSE  ――  converts a WFOMC problem to a *grounded* CNF file
# ----------------------------------------------------------------------
def convert_to_cnf(problem: WFOMCProblem, output_path: str):
    """
    Convert a WFOMCProblem instance to a grounded CNF file that can be
    handled by sharpSAT.

    **Fix:** counting-quantifier constraints (∃_{=k} / ∃_{r mod k})
    now ground *all* free variables (outer X and inner Y) before being
    encoded, removing the under-constrained bug present previously.
    """
    domain = problem.domain
    expr = sympy.true  # global conjunction we will gradually build
    atom_to_digit = {}  # AtomicFormula -> int  (positive ID)
    atomsym_to_digit = {}  # sympy.Symbol   -> int (positive ID)

    # ------------------------------------------------------------------
    # Helper – register fresh atoms with the DIMACS numbering tables
    # ------------------------------------------------------------------
    def _register_atoms(qf: QFFormula):
        for atom in qf.atoms():
            if atom not in atom_to_digit:
                idx = len(atom_to_digit) + 1
                atom_to_digit[atom] = idx
                atomsym_to_digit[atom.expr] = idx

    # ---------- 1.  Ground the quantifier-free *base* part ----------
    uni_formula = extract_qf_formula(copy.deepcopy(problem.sentence.uni_formula))
    for e1, e2 in product(domain, repeat=2):
        grounded = uni_formula.substitute({X: e1, Y: e2}) & \
                   uni_formula.substitute({X: e2, Y: e1})
        expr &= sympy.to_cnf(grounded.expr)
        _register_atoms(grounded)

    # ---------- 2.  Ground pure ∃Y … “extension” formulas ----------
    ext_formulas = [extract_qf_formula(copy.deepcopy(f.quantified_formula))
                    for f in problem.sentence.ext_formulas]

    for e1 in domain:
        # outer variable X is fixed to e1
        for ext_formula in ext_formulas:
            disj = sympy.false
            for e2 in domain:
                grounded = ext_formula.substitute({X: e1, Y: e2})
                disj |= grounded.expr
                _register_atoms(grounded)
            expr &= disj

    # ---------- 3.  Encode counting quantifiers ----------
    cnt_formulas = copy.deepcopy(problem.sentence.cnt_formulas)

    for e1 in domain:  # ∀X ...
        for cnt_formula in cnt_formulas:
            q_scope = cnt_formula.quantified_formula.quantifier_scope
            var_y = q_scope.quantified_var  # the Y in ∃_{=k} Y …
            comparator = q_scope.comparator  # '=' or 'mod'
            param = q_scope.count_param  # k  or  (r,k)
            inner_qf = cnt_formula.quantified_formula.quantified_formula

            # --- locate the free outer variable (should be exactly one) ---
            free_vars = inner_qf.vars() - {var_y}
            if len(free_vars) != 1:
                raise ValueError("Counting formula should have exactly one free variable")
            (var_x,) = free_vars

            # --- build the cardinality/parity constraint over literals E(e1,·) ---
            if comparator == '=':
                k_val = param
                clause_expr = sympy.false
                for combo in combinations(domain, k_val):
                    sub = sympy.true
                    # selected Y's are true
                    for y in combo:
                        g = inner_qf.substitute({var_x: e1, var_y: y})
                        sub &= g.expr
                        _register_atoms(g)
                    # the rest are false
                    for y in set(domain) - set(combo):
                        g = inner_qf.substitute({var_x: e1, var_y: y})
                        sub &= sympy.Not(g.expr)
                        _register_atoms(g)
                    clause_expr |= sub
                clause_expr = sympy.simplify_logic(clause_expr, form='cnf')
                expr &= clause_expr

            elif comparator == 'mod':
                r, k_mod = param
                clause_expr = sympy.false
                for n in range(len(domain) + 1):
                    if n % k_mod != r:
                        continue
                    for combo in combinations(domain, n):
                        sub = sympy.true
                        for y in combo:
                            g = inner_qf.substitute({var_x: e1, var_y: y})
                            sub &= g.expr
                            _register_atoms(g)
                        for y in set(domain) - set(combo):
                            g = inner_qf.substitute({var_x: e1, var_y: y})
                            sub &= sympy.Not(g.expr)
                            _register_atoms(g)
                        clause_expr |= sub
                clause_expr = sympy.simplify_logic(clause_expr, form='cnf')
                expr &= clause_expr

            else:
                raise NotImplementedError(f"Unsupported comparator {comparator!r}")

    # ---------- 4.  Final CNF conversion & DIMACS dump ----------
    expr = sympy.to_cnf(expr)

    # ensure every atom got a DIMACS id
    for atom in QFFormula(expr).atoms():
        if atom not in atom_to_digit:
            idx = len(atom_to_digit) + 1
            atom_to_digit[atom] = idx
        if atom.expr not in atomsym_to_digit:
            atomsym_to_digit[atom.expr] = atom_to_digit[atom]

    clauses = expr.args if isinstance(expr, sympy.And) else [expr]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"c atoms: {' '.join(map(str, atom_to_digit.keys()))}\n")
        f.write(f"p cnf {len(atom_to_digit)} {len(clauses)}\n")
        for clause in clauses:
            clause_str = ''
            atoms = clause.args if isinstance(clause, sympy.Or) else [clause]
            for atom in atoms:
                if isinstance(atom, sympy.Symbol):
                    if atom not in atomsym_to_digit:
                        raise KeyError(f"Unmapped atom: {atom}")
                    clause_str += f"{atomsym_to_digit[atom]} "
                elif isinstance(atom, sympy.Not):
                    sym = ~atom
                    if sym not in atomsym_to_digit:
                        raise KeyError(f"Unmapped NOT atom: {sym}")
                    clause_str += f"{-atomsym_to_digit[sym]} "
                else:
                    raise RuntimeError(f"Unknown atom type: {atom}")
            f.write(clause_str.strip() + " 0\n")
    logger.info("CNF written to %s", output_path)


# ----------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    loglevel(20)  # INFO
    args = parse_args()
    problem = get_problem(args.input, args.domain_size)
    sentence_base = os.path.basename(args.input)
    output_file = os.path.join("check", f"{sentence_base[:-7]}_{args.domain_size}.cnf")
    convert_to_cnf(problem, output_file)
