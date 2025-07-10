"""
用途：将 CNF 计算model counting
使用：修改文件目录，然后运行这个文件，
使用 pysat 来计算模型数量，同时将sharpSAT备用
"""

import subprocess
from logzero import logger
from pysat.formula import CNF
from pysat.solvers import Solver
import pandas as pd


def count_models_with_sharpSAT(cnf_path):
    # ✅ Step 1: 用 PySAT 加载并清洗 CNF
    cleaned_path = cnf_path.replace('.cnf', '_cleaned.cnf')
    try:
        cnf = CNF(from_file=cnf_path)
        cnf.to_file(cleaned_path)
    except Exception as e:
        print(f"❌ 读取或写入 CNF 文件失败: {e}")
        return None

    try:
        result = subprocess.run(
            ["/usr/local/bin/sharpSAT", cleaned_path],
            capture_output=True,  # 捕获命令的标准输出和标准错误。
            text=True  # 输出结果为字符串形式，而不是字节流。
        )

        # logger.info("=== 标准输出 ===","\n",result.stdout) # 用于输出程序正常执行时产生的信息。
        print(result.stdout)
        if result.returncode != 0:
            print("错误信息：")
            print(result.stderr)


    except FileNotFoundError:
        print("❌ sharpSAT 不在指定路径")
        return None


def count_with_pysat(path):
    cnf = CNF(from_file=path)
    count = 0
    with Solver(bootstrap_with=cnf) as s:
        for model in s.enum_models():
            count += 1
    # print("Model count:", count)
    return count


if __name__ == '__main__':
    # count_models_with_sharpSAT(path)  model counting为1的时候，sharpSAT会报错，但是其余情况正常使用，这里作为pysat的备用，适用于大规模

    # path = "/root/pycharm_workspace/modk/check/2-regular-graph-sc2_2.cnf"
    # path = "/root/pycharm_workspace/modk/check/0mod2-regular-graph-sc2_6.cnf"

    df = pd.DataFrame(columns=['domain_size', 'model_count'])
    for domain_size in range(1, 7):
        path = f"/root/pycharm_workspace/modk/check/0mod2-regular-graph-sc2_{domain_size}.cnf"
        mc = count_with_pysat(path)
        new_row = pd.DataFrame([{'domain_size': domain_size, 'model_count': mc}])
        df = pd.concat([df, new_row], ignore_index=True)
    print(df)
    df.to_csv("0mod2-regular-graph-sc2.csv", index=False)
