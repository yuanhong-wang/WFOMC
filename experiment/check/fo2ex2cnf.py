import os
import copy
import logzero
from logzero import logger
import argparse
from itertools import product, permutations
import sympy
from pysat.card import CardEnc, EncType

from wfomc import parse_input
from wfomc.problems import WFOMCProblem
from wfomc.fol.syntax import Const, X, Y, QFFormula, AtomicFormula, Pred



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a first-order logic sentence to CNF',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True, help='sentence file')
    parser.add_argument('--linearencode', '-e', type=int, required=False, help='linear order and predcessor encode method 0-disable 1-hard encode 2-TODO first order encode',default=0 )
    args = parser.parse_args()
    return args

def generate_leq_pred_axioms(domain, atom_to_digit):

    leq_pred_clauses = []
    consts = sorted(list(domain), key=lambda c: str(c))
   
    for atom, var_id in atom_to_digit.items():
        pred_name = atom.pred.name
        if pred_name == "LEQ":
            left, right = atom.args
            if consts.index(left) <= consts.index(right):
                leq_pred_clauses.append([var_id])   # True
            else:
                leq_pred_clauses.append([-var_id])  # False

        elif pred_name == "PRED1": #PRED in new_encode
            left, right = atom.args
            li, ri = consts.index(left), consts.index(right)
            if ri == li + 1:  
                leq_pred_clauses.append([var_id])
            else:
                leq_pred_clauses.append([-var_id])

        elif pred_name == "CIRCULAR_PRED": #CIRPRED in new_encode
            left, right = atom.args
            li, ri = consts.index(left), consts.index(right)
            if ri == li + 1 or (ri == 0 and li == len(consts)-1):  
                leq_pred_clauses.append([var_id])
            else:
                leq_pred_clauses.append([-var_id])

    return leq_pred_clauses

            
def generate_xyz_leq_expr(domain)->sympy.Expr:
    
    leqatom:AtomicFormula=None
    for atom, var_id in atom_to_digit.items():
        if atom.pred.name == "LEQ": 
            leqatom = atom
            break
    if leqatom is None or len(domain) < 2:
        return []
    
    xyz_expr: sympy.Expr = sympy.true
    leq_pred = leqatom.pred
    
    for x in domain: 
        #VX: ~LEQ(X,X)
        #LEQ(x,x)
        ground_atom = AtomicFormula(leq_pred, (x, x), True)
        xyz_expr = sympy.And(xyz_expr, ground_atom.expr)

    for x, y, z in product(domain, repeat=3): 
        #VXVYVZ: LEQ(X,Y) & LEQ(Y,Z) -> LEQ(X,Z)
        # ~( (LEQ(x,y) & LEQ(y,z) ) & ~LEQ(x,z) )
        # ~(LEQ(x,y) & LEQ(y,z)) | LEQ(x,z)
        # ~LEQ(x,y) | ~LEQ(y,z) | LEQ(x,z)

        ground_atom1 = AtomicFormula(leq_pred, (x, y), False)  
        ground_atom2 = AtomicFormula(leq_pred, (y, z), False) 
        ground_atom3 = AtomicFormula(leq_pred, (x, z), True) 
        ground_atom = sympy.Or(ground_atom1.expr, ground_atom2.expr, ground_atom3.expr)
        xyz_expr = sympy.And(xyz_expr, ground_atom)                   

    for x, y in permutations(domain, 2):
        # VXVY: LEQ(X,Y) | LEQ(Y,X)
        ground_atom1 = AtomicFormula(leq_pred, (x, y), True)  
        ground_atom2 = AtomicFormula(leq_pred, (y, z), True) 
        ground_atom = sympy.Or(ground_atom1.expr, ground_atom2.expr)
        xyz_expr = sympy.And(xyz_expr, ground_atom)                   

    return xyz_expr


if __name__ == "__main__":
    logzero.loglevel(logzero.INFO)
    args = parse_args()
    sentence_dir = os.path.dirname(args.input)
    sentence_base = os.path.basename(args.input)
    problem = parse_input(args.input)
    leq_support = args.linearencode

    # remove the quantifier
    uni_formula: QFFormula = copy.deepcopy(problem.sentence.uni_formula)
    ext_formulas: list[QFFormula] = copy.deepcopy(problem.sentence.ext_formulas)
    

    while not isinstance(uni_formula, QFFormula):
        uni_formula = uni_formula.quantified_formula
    for i in range(len(ext_formulas)):
        ext_formulas[i] = ext_formulas[i].quantified_formula.quantified_formula


    atom_to_digit: dict[AtomicFormula, int] = {}
    atomsym_to_digit: dict[sympy.Symbol, int] = {}
    expr: sympy.Expr = sympy.true
    
    domain = problem.domain
    for (e1, e2) in product(domain, repeat=2):
        ground_uni_formula: QFFormula = uni_formula.substitute({X: e1, Y: e2}) & uni_formula.substitute({X: e2, Y: e1})
        # cnf_formula = sympy.to_cnf(ground_uni_formula.expr, simplify=True)
        cnf_formula = sympy.to_cnf(ground_uni_formula.expr)
        expr = sympy.And(expr, cnf_formula)

        for atom in ground_uni_formula.atoms():
            if atom not in atom_to_digit:
                atom_to_digit[atom] = len(atom_to_digit)+1
                atomsym_to_digit[atom.expr] = len(atomsym_to_digit)+1
    
    for e1 in domain:
        for ext_formula in ext_formulas:
            ext_expr = sympy.false
            for e2 in domain:
                ground_ext_formula = ext_formula.substitute({X: e1, Y: e2})
                ext_expr = sympy.Or(ext_expr, ground_ext_formula.expr)
                for atom in ground_ext_formula.atoms():
                    if atom not in atom_to_digit:
                        atom_to_digit[atom] = len(atom_to_digit)+1
                        atomsym_to_digit[atom.expr] = len(atomsym_to_digit)+1
            expr = sympy.And(expr, ext_expr)


#============ leq support 2 ================
    if leq_support == 2:
        print("WARNING: There might be bugs in encode2")
        expr = sympy.And(expr, generate_xyz_leq_expr(domain))

#============ get_clause ================
    expr = sympy.to_cnf(expr)
    cnf_clause_list = []
    for clause in expr.args:
        clause_str = ""
        if isinstance(clause, sympy.Not):
            clause_str += str(-atomsym_to_digit[~clause]) + " "
        elif isinstance(clause, sympy.Symbol):
            clause_str += str(atomsym_to_digit[clause]) + " "
        else:
            for atom in clause.args:
                if isinstance(atom, sympy.Symbol):
                    clause_str += str(atomsym_to_digit[atom]) + " "
                elif isinstance(atom, sympy.Not):
                    clause_str += str(-atomsym_to_digit[~atom]) + " "
                else:
                    raise RuntimeError(f'Unknown atom type: {atom}')
    
        line = str(clause_str.strip()) + " 0\n"
        cnf_clause_list.append(line)

#============ unary evidence ===============
    if problem.unary_evidence is not None:
        unary_evidence: set[AtomicFormula] = problem.unary_evidence
        ue_atoms=[] 

        for atom in unary_evidence:
            if atom.positive:
                ue_atoms.append((atom, True))
            else:
                ue_atoms.append((~atom, False))

        ue_clauses = []
        for item in ue_atoms:
            atom=item[0]
            pos=item[1]
            if atom not in atom_to_digit:
                atom_to_digit[atom] = len(atom_to_digit) + 1
                atomsym_to_digit[atom.expr] = atom_to_digit[atom]

            lit = atom_to_digit[atom] if pos else -atom_to_digit[atom]
            ue_clauses.append([lit])

        for clause in ue_clauses:
            line = " ".join(map(str, clause)) + " 0\n"
            cnf_clause_list.append(line)

# #============ leq support 1 hard encode ================
    if leq_support == 1:
        leq_clauses=generate_leq_pred_axioms(domain, atom_to_digit)
        for clause in leq_clauses:
            line = " ".join(map(str, clause)) + " 0\n"
            cnf_clause_list.append(line)

#============cc===========
    if problem.cardinality_constraint is not None:
        constraints = problem.cardinality_constraint.constraints # 获取所有约束的列表。每个约束通常包含谓词、操作符和界限。
        cc_clauses = [] # 初始化一个列表，用于存放从基数约束转换来的所有CNF子句。
        for pred_map, op, bound in constraints: # 遍历每一个约束。每个约束是一个元组，如 ({Pred:"P"}, "<=", 5)。
            for pred, coeff in pred_map.items(): # 遍历约束中涉及的谓词。这段代码的结构暗示每个约束只处理一个谓词。
                pred_name = str(pred) # 获取谓词的名称，例如 "P"。
                k = int(bound)  # 获取约束的数值界限，例如 5。

                
                vars = [v for ksym, v in atomsym_to_digit.items() if pred_name in str(ksym)] # 这一行非常关键：它从已经建立的变量映射中，找出所有与当前谓词相关的变量。# 例如，如果谓词是 "P"，它会找到 P(c1), P(c2), ... 等所有基化原子对应的整数变量。
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
                            ccatom:AtomicFormula= AtomicFormula( Pred('CC'+str(len(atom_to_digit)+1), 1), 'c',True)
                            atom_to_digit[ccatom] = len(atom_to_digit)+1 # 将新原子和它的新整数ID添加到全局映射中。
                            atomsym_to_digit[ccatom.expr] = len(atomsym_to_digit)+1
                            ignore_atom.append(len(atom_to_digit)) # 将这个新的辅助变量ID添加到ignore_atom列表中，避免重复处理。

                            for j in modify_clauses: # 再次遍历所有子句，将pysat临时分配的辅助变量ID替换为我们刚刚注册的全局唯一ID。
                                for k_idx, k in enumerate(j): 
                                    if abs(i) == abs(k):
                                        j[k_idx] = len(atom_to_digit) if k > 0 else -len(atom_to_digit)

                #print('after:',len(atom_to_digit), modify_clauses)

                cc_clauses.extend(modify_clauses) # 将处理完（即所有变量ID都已全局注册）的子句添加到cc_clauses列表中。

        for clause in cc_clauses: # 遍历所有为基数约束生成的子句。
            line = " ".join(map(str, clause)) + " 0\n" # 将每个子句格式化为DIMACS CNF标准的一行（整数用空格隔开，以0结尾）。
            cnf_clause_list.append(line) # 将格式化后的行添加到总的CNF子句列表中。
 


    cnf_clause_str="".join(cnf_clause_list)
    #print(cnf_clause_str)

    klist = list(atom_to_digit.keys())
    vlist = list(atom_to_digit.values())
    kstr = f'c {" ".join([str(k) for k in klist])}\n'
    vstr = f'c {" ".join([str(v) for v in vlist])}\n'

    cnf_file_path = os.path.join(sentence_dir, f'{os.path.splitext(sentence_base)[0] }.cnf')

    cnf_file = open(cnf_file_path, 'w')
    cnf_file.write(kstr)
    cnf_file.write(vstr)

    cnf_file.write(f"p cnf {len(atom_to_digit)} {len(cnf_clause_list)}\n")
    cnf_file.write(cnf_clause_str)

    cnf_file.close()
    logger.info('CNF file written to %s', cnf_file_path)

