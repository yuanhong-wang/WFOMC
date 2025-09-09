#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CNF模型计数工具

文件路径：check/count_from_cnf.py

功能概述：
本脚本提供了两种CNF(合取范式)模型数量(Model Counting)的计算方法：
1. PySAT库实现的枚举计算（适用于中小规模问题）
2. 外部sharpSAT工具的高性能计算（适用于大规模问题）

主要功能：
- 提供Python原生的模型计数功能(count_with_pysat)
- 集成sharpSAT求解器作为高性能备选方案(count_models_with_sharpSAT)
- 支持批量测试不同域大小(domain size)的模型计数
- 自动生成CSV格式的测试结果报告
- 自动清洗CNF文件格式确保兼容性

使用方法：
1. 准备工作：
   - 安装依赖：pysat, pandas, logzero
   - (可选)安装sharpSAT并配置路径(/usr/local/bin/sharpSAT)
   - 准备CNF测试文件(命名格式:*_<domain_size>.cnf)

2. 配置说明：
   - 修改主程序中的CNF文件路径模板：
     path = f"path/to/your/file_{domain_size}.cnf"
   - 调整测试的域大小范围(range参数)

3. 运行命令：
   python count_from_cnf.py

4. 输出结果：
   - 控制台输出每个域大小的模型计数
   - 生成CSV结果文件：0mod2-regular-graph-sc2.csv

注意事项：
- 默认使用PySAT计算，取消注释可启用sharpSAT
- sharpSAT在domain_size=1时可能报错，已做异常处理
- 建议小规模测试用PySAT，大规模用sharpSAT
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

        # logger.info("=== 标准输出 ===","\n",results.stdout) # 用于输出程序正常执行时产生的信息。
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
