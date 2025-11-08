from wfomc import wfomc, parse_input, Algo, Const
from wfomc.parser.wfomcs_parser import parse as wfomcs_parse
from sympy import sympify, Symbol, Poly
import logzero, logging
logging.disable(logging.CRITICAL)
import csv
model_str = r"""
\forall X: (\exists_{1 mod 2} Y: (B(X, Y))) &
\forall X: (~E(X,X)) &
\forall X: (\forall Y: (E(X,Y) -> E(Y,X))) &
\exists_{=m} X: (Odd(X)) &
\exists_{=1} X: (U(X)) &
\forall X: (P(X) <-> (~Odd(X) & A(X) & C(X))) &
\forall X: (\forall Y: (P(X) & B(X,Y) -> U(Y))) &
\forall X: (\forall Y:(~P(X) -> (B(X,Y) <-> E(X,Y)))) &
\forall X: (Odd(X) | A(X)) &
\forall X: (A(X) | C(X)) 

n = 3
1 -1 C
|E| = 6
"""
from sympy import S
from wfomc.utils import expand

m = 2
k = 2
max_n = 10
# 3. 打开文件并创建 csv writer
output_file = '/home/sunshixin/pycharm_workspace/WFOMC/experiment/check/nmk_experiment/nmk.csv'
with open(output_file, 'a', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['n', 'm', 'k'])  # 写入表头
    
    for n in range(1, max_n + 1): #
        # print(f"n={n}")
        m_list = list(range(0, n+1, 2))  # m取0,2,4,...,n
        for m in m_list:
            model = model_str.replace('=m', f'={m}')
            # print("Modified model:")
            # print(model)
            problem = wfomcs_parse(model)
            # print("Parsed problem:", problem)
            problem.domain = {Const(f'd{i}') for i in range(n)}
            res = wfomc(problem, algo=Algo.DR)  # Divide by nC1 = n
            print("Raw result:", res)
            # 获取系数，
            expr = sympify(str(res))
            x0 = Symbol('x0')

            # 3. 将表达式转换为多项式对象
            p = Poly(expr, x0)

            # 4. 获取多项式的次数 (即最高次幂/指数)
            highest_degree = p.degree()
            # print(p)
            # print(highest_degree)

            # 5. 获取最高次项的系数 (leading coefficient)
            leading_coeff = p.LC()

            # new_degree = list(range(highest_degree + 1))
            if isinstance(highest_degree, int):
                new_degree = list(range(highest_degree + 1))
            elif highest_degree == S.NegativeInfinity or str(highest_degree) == "-oo":
                # 负无穷时，返回空列表或做特殊处理
                new_degree = []
                
            # print("degrees:", new_degree)
            k = p.all_coeffs()[::-1][::2]  # 从低次到高次排列
            # print("coefficients:", coeffs)
            k_str = ",".join(map(str, k))
            writer.writerow([n, m, k_str])
            print(f"{n}, {m}, {k_str}")
            f_out.flush()
print("finished!")    

    

