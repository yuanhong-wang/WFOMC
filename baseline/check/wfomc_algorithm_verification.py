import csv
from multiprocessing import Process, Queue

import psutil
import time

from matplotlib import pyplot as plt
from wfomc import wfomc, parse_input, Const, Algo

import os
import subprocess
from datetime import datetime
from fo2_to_cnf_converter import fo2_to_cnf_count
import fo2_to_cnf_converter
import pandas as pd
from pathlib import Path
from pathlib import Path
from typing import List, Union, Iterable
import logging
from wfomc import Algo
from logzero import logger, logfile
import logzero
from tqdm import tqdm

DIR_PATH = Path(__file__).parent  # 获取项目根目录


class ExperimentConfig:
    """实验配置类"""
    # 路径配置
    global DIR_PATH
    MODELS_PATH = os.path.join(DIR_PATH, "models")  # 输入例子的路径
    RESULTS_PATH = os.path.join(DIR_PATH, "results")  # 存储实验结果的路径
    TIMEOUT_SECONDS = 100  # 超市时间，5分钟超时
    # 创建结果目录
    os.makedirs(RESULTS_PATH, exist_ok=True)
    # 实验参数，示例，后面可以修改
    DOMAIN_SIZES = list(range(2, 70, 2))
    MODELS = {
    }
    # 算法配置，示例，后面可以修改
    ALGORITHMS = ["dr", "recursive", "fast"]
    # 实验组配置：每组包含特定的域大小范围、算法列表和模型列表
    EXPERIMENT_GROUPS = dict()


class MyTools:
    @staticmethod
    def run_with_timeout(func, args=(), timeout=100):
        q = Queue()  # # 创建一个进程间通信队列，用于获取子进程的执行结果

        def wrapper(q):  # 定义包装函数，用于在子进程中执行目标函数并将结果放入队列
            res = func(*args)  # 执行传入的函数，传入参数并获取结果
            q.put(res)  # 将结果放入队列供主进程获取

        p = Process(target=wrapper, args=(q,))  # 创建子进程，执行包装函数并传入队列参数
        p.start()  # 启动子进程开始执行
        p.join(timeout)  # 等待子进程执行完成，最多等待timeout秒
        if p.is_alive():  # 检查子进程是否仍在运行（超时情况）
            p.terminate()  # 如果子进程仍在运行，强制终止它
            p.join()  # 等待子进程完全退出并清理资源
            return "TIMEOUT"  # 返回超时标识
        return q.get() if not q.empty() else "ERROR"  # 如果子进程已完成，从队列中获取结果并返回，如果队列为空则返回错误标识

    @staticmethod
    def generate_results_filename(subdir, name):
        """生成带时间戳的结果文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间并格式化为字符串
        result_dir = os.path.join(ExperimentConfig.RESULTS_PATH, subdir)
        if os.path.exists(result_dir) is False:
            os.makedirs(result_dir)  # 如果结果目录不存在，则创建它
        return os.path.join(
            result_dir,
            f"baseline_{name}_{timestamp}.csv"  # 生成的时间戳
        )

    @staticmethod
    def write_result(writer, result_data, csvfile):
        """即时写入并落盘"""
        writer.writerow(result_data)  # 将结果数据写入CSV文件的一行
        csvfile.flush()  # 强制将缓冲区的数据写入磁盘
        os.fsync(csvfile.fileno())  # 强制操作系统将数据从内核缓冲区写入磁盘，确保数据持久化

    @staticmethod
    def print_result(csv_file):
        RESULTS_PATH = ExperimentConfig.RESULTS_PATH  # 获取实验结果路径
        csv_path = os.path.join(RESULTS_PATH, csv_file)  # 构造CSV文件的完整路径
        # 1. 读取 CSV
        data = pd.read_csv(csv_file)
        data = data[data['status'] != "timeout"]  # 过滤掉超时数据
        formula = data['formula'].iloc[0]  # 获取第一个公式

        # 2. 拆分三种算法
        recursive_data = data[data['algorithm'].str.contains("dr", case=False)]
        ganak_data = data[data['algorithm'].str.contains("ganak", case=False)]
        approx_data = data[data['algorithm'].str.contains("approxmc", case=False)]

        # 3. 处理数据
        ouralgo_domain = recursive_data['domain_size']
        ouralgo_value = recursive_data['result']

        ganak_domain = ganak_data['domain_size']
        ganak_values = ganak_data['result']

        approx_domain = approx_data['domain_size']
        approx_values = approx_data['result']
        # 确保数值列是数值类型，而不是字符串
        ouralgo_value = pd.to_numeric(ouralgo_value, errors='coerce') # 参数 errors='coerce'：当遇到无法转换为数值的数据时（如字符串 "TIMEOUT"），将其转换为 NaN（Not a Number），而不是抛出异常
        ganak_values = pd.to_numeric(ganak_values, errors='coerce')
        approx_values = pd.to_numeric(approx_values, errors='coerce')
        # 删除NaN值
        ouralgo_domain = ouralgo_domain[ouralgo_value.notna()]
        ouralgo_value = ouralgo_value[ouralgo_value.notna()]
        ganak_domain = ganak_domain[ganak_values.notna()]
        ganak_values = ganak_values[ganak_values.notna()]
        approx_domain = approx_domain[approx_values.notna()]
        approx_values = approx_values[approx_values.notna()]
        # 确保所有值都大于等于1
        mask_ouralgo = (ouralgo_value >= 1) & (ouralgo_domain<=10)
        ouralgo_domain = ouralgo_domain[mask_ouralgo]
        ouralgo_value = ouralgo_value[mask_ouralgo]
        # 获取ouralgo_domain的范围作为基准
        common_domain = ouralgo_domain
        # 根据common_domain过滤ganak数据
        mask_ganak = (ganak_values >= 1) & (ganak_domain.isin(common_domain))
        ganak_domain = ganak_domain[mask_ganak]
        ganak_values = ganak_values[mask_ganak]
        # 根据common_domain过滤approx数据
        mask_approx = (approx_values >= 1) & (approx_domain.isin(common_domain))
        approx_domain = approx_domain[mask_approx]
        approx_values = approx_values[mask_approx]


        # 这里假设没有误差，构造一个 ±5% 区间
        if not approx_values.empty:
            # 这里假设没有误差，构造一个 ±5% 区间
            approx_lower = approx_values * (1 - epsilon)
            approx_upper = approx_values * (1 + epsilon)


        # 4. 画图
        plt.close('all')
        plt.figure(figsize=(12, 8))

        if not ouralgo_value.empty:
            plt.plot(ouralgo_domain, ouralgo_value, label="OurAlgo", color='#ff7f0e', linewidth=3)
        if not ganak_values.empty:
            plt.scatter(ganak_domain, ganak_values, color='#1f77b4', marker='s', s=50, label="Ganak (Exact)")
        if not approx_values.empty:
            plt.errorbar(approx_domain, approx_values,
                         yerr=[approx_values - approx_lower, approx_upper - approx_values],
                         fmt='o', color='#2ca02c', ecolor='#98df8a', elinewidth=2, capsize=4, label="ApproxMC (95% CI)")

        # 5. 坐标轴 & 样式
        plt.yscale("log")
        plt.xlabel("Domain Size", fontsize=20)
        plt.ylabel("Model Count (log scale)", fontsize=20)
        # plt.title("Model Counting Comparison", fontsize=16)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        # 假设你已经有了这三个数据列表
        max_ouralgo = max(ouralgo_value) if not ouralgo_value.empty else 0
        max_ganak = max(ganak_values) if not ganak_values.empty else 0
        max_approx = max(approx_values) if not approx_values.empty else 0

        # 找到这三个列表中的最大值
        max_value = max(max_ouralgo, max_ganak, max_approx)

        # 适当放大 max_value
        max_value *= 1.1  # 放大 10%

        # 或者选择一个更合理的数量级
        if max_value < 10:
            max_value = 10
        elif max_value < 100:
            max_value = 100
        elif max_value < 1000:
            max_value = 1000
        plt.ylim(0.1, max_value)

        # 6. 保存
        os.makedirs(RESULTS_PATH, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        pdf_path = os.path.join(RESULTS_PATH, f"{formula}_check.pdf")
        png_path = os.path.join(RESULTS_PATH, f"{formula}_check.png")

        plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight')
        plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
        print(f"图表已保存至：\n{pdf_path}\n{png_path}")
        plt.show()


def run_single_experiment(formula_name, file, domain_size, algo):
    """运行单个实验（多进程+超时强制终止）"""
    ## 准备问题
    file_path = os.path.join(ExperimentConfig.MODELS_PATH, file)  # 构造模型文件的完整路径
    problem = parse_input(file_path)  # 解析模型文件，获取问题定义
    problem.domain = {Const(f'd{i}') for i in range(domain_size)}  # 根据指定的域大小构造域集合

    # 记录开始时间和内存
    process = psutil.Process(os.getpid())  # 获取当前进程
    start_mem = process.memory_info().rss  # 记录开始内存 (单位: bytes)
    start_time = time.time()

    ## 调用多进程执行 wfomc
    res = MyTools.run_with_timeout(wfomc, args=(problem, algo), timeout=ExperimentConfig.TIMEOUT_SECONDS)  # 使用带超时控制的方式执行wfomc函数

    # 结束记录
    end_time = time.time()
    elapsed = end_time - start_time  # 计算执行时间 (单位: 秒)
    end_mem = process.memory_info().rss  # 结束内存 (bytes)
    mem_used_bytes = max(end_mem - start_mem, 0)  # 防止负数
    mem_used_kb = mem_used_bytes / 1024
    mem_used_mb = mem_used_bytes / (1024 * 1024)

    ## 记录结果
    status = "timeout" if res == "TIMEOUT" else "completed"  # 根据执行结果判断是否超时
    if status == "completed":  # 执行成功，记录INFO级别日志
        logger.info(
            f"[OK] 公式:{formula_name}, 域:{domain_size}, 算法:{algo}, 结果:{res}, 耗时:{elapsed:.2f}s, 内存字节:{mem_used_bytes}字节, 内存KB:{mem_used_kb}KB, 内存:{mem_used_mb}MB, 执行状态:{status}")
    else:  # 执行超时，记录WARNING级别日志
        logger.warning(
            f"[TIMEOUT] 公式:{formula_name}, 域:{domain_size}, 算法:{algo}, 结果:{res}, 耗时:{elapsed:.2f}s, 内存字节:{mem_used_bytes}字节, 内存KB:{mem_used_kb}KB, 内存:{mem_used_mb}MB, 执行状态:{status}")

    ## 构建结果
    result_data = {  # 构建结果，将实验结果组织成字典格式
        'timestamp': datetime.now().isoformat(),  # 当前时间戳
        'formula': formula_name,  # 公式名称
        'domain_size': domain_size,  # 域大小
        'algorithm': str(algo),  # 使用的算法
        'result': res,  # 计算结果
        # 'time_sec': elapsed if status == "completed" else ExperimentConfig.TIMEOUT_SECONDS,  # 执行时间
        # 'memory_bytes': mem_used_bytes,  # 内存使用量（字节）
        # 'memory_kb': round(mem_used_kb, 2),  # 内存使用量（KB，保留2位小数）
        # 'memory_mb': round(mem_used_mb, 4),  # 内存使用量（MB，保留4位小数）
        'status': status  # 执行状态
    }

    # 返回实验结果数据
    return result_data


def cnf_pysat(input_file, domain_size):
    return fo2_to_cnf_count(input_file, domain_size, "pysat")  # 调用fo2_to_cnf_count函数


def cnf_ganak(input_file, domain_size):
    res = MyTools.run_with_timeout(fo2_to_cnf_count, args=(input_file, domain_size, "ganak"), timeout=ExperimentConfig.TIMEOUT_SECONDS)
    # res = fo2_to_cnf_count(input_file, domain_size, "ganak")  # 调用fo2_to_cnf_count函数
    status = "timeout" if res == "TIMEOUT" else "completed"  # 根据执行结果判断是否超时
    result_data = {  # 构建结果，将实验结果组织成字典格式
        'timestamp': datetime.now().isoformat(),  # 当前时间戳
        'formula': os.path.splitext(os.path.basename(input_file))[0],  # 公式名称
        'domain_size': domain_size,  # 域大小
        'algorithm': "ganak",  # 使用的算法
        'result': res,  # 计算结果
        # 'time_sec': elapsed if status == "completed" else ExperimentConfig.TIMEOUT_SECONDS,  # 执行时间
        # 'memory_bytes': mem_used_bytes,  # 内存使用量（字节）
        # 'memory_kb': round(mem_used_kb, 2),  # 内存使用量（KB，保留2位小数）
        # 'memory_mb': round(mem_used_mb, 4),  # 内存使用量（MB，保留4位小数）
        'status': status  # 执行状态
    }
    return result_data  # 返回实验结果数据


def cnf_approxmc(input_file, domain_size):
    global epsilon,  delta
    res = MyTools.run_with_timeout(fo2_to_cnf_count, args=(input_file, domain_size, "approxmc", epsilon, delta), timeout=ExperimentConfig.TIMEOUT_SECONDS)
    # res = fo2_to_cnf_count(input_file, domain_size, )  # 调用fo2_to_cnf_count函数
    status = "timeout" if res == "TIMEOUT" else "completed"  # 根据执行结果判断是否超时
    result_data = {  # 构建结果，将实验结果组织成字典格式
        'timestamp': datetime.now().isoformat(),  # 当前时间戳
        'formula': os.path.splitext(os.path.basename(input_file))[0],  # 公式名称
        'domain_size': domain_size,  # 域大小
        'algorithm': "approxmc",  # 使用的算法
        'result': res,  # 计算结果
        # 'time_sec': elapsed if status == "completed" else ExperimentConfig.TIMEOUT_SECONDS,  # 执行时间
        # 'memory_bytes': mem_used_bytes,  # 内存使用量（字节）
        # 'memory_kb': round(mem_used_kb, 2),  # 内存使用量（KB，保留2位小数）
        # 'memory_mb': round(mem_used_mb, 4),  # 内存使用量（MB，保留4位小数）
        'status': status,  # 执行状态
        # "dr_within_approx_confidence"
    }
    return result_data  # 返回实验结果数据


def run_experiment():
    # 定义CSV文件的列名
    fieldnames = ['timestamp', 'formula', 'domain_size', 'algorithm', 'result', 'status', "dr_within_approx_confidence"]

    ## 遍历所有实验组
    for group in ExperimentConfig.EXPERIMENT_GROUPS:
        group_name = group["name"]
        domain_sizes = group["domain_sizes"]
        algorithms = group["algorithms"]
        models = group["models"]
        print(f"\n开始执行实验组: {group_name}，域大小范围: {domain_sizes}，算法列表: {algorithms}，模型列表: {list(models.keys())}")
        # 获取总的迭代次数
        total_iterations = len(ExperimentConfig.MODELS) * len(ExperimentConfig.DOMAIN_SIZES) * len([a for a in Algo if str(a) in ExperimentConfig.ALGORITHMS])  # 计算总共需要执行的实验次数：模型数 × 域大小数 × 算法数
        with tqdm(total=total_iterations, desc="Overall Progress") as pbar:  # 创建一个总体进度条，显示实验整体进度
            ## 1 遍历所有模型（公式名称和文件名）
            for formula_name, file in models.items():
                result_file_path = MyTools.generate_results_filename(group_name, formula_name)  # 为当前模型生成结果文件名
                with open(result_file_path, 'w', newline='') as csvfile:  # 打开结果文件准备写入
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # 创建CSV字典写入器
                    writer.writeheader()  # 写入CSV文件头部（列名）
                    ## 2 遍历所有需要测试的算法
                    for algo in [a for a in Algo if str(a) in algorithms]:
                        dr_timeout = False
                        ganak_timeout = False
                        approxmc_timeout = False
                        for domain_size in domain_sizes:  ## 遍历所有域大小

                            if not dr_timeout:
                                result_dr = run_single_experiment(formula_name, file, domain_size, algo)
                                print(result_dr)
                                MyTools.write_result(writer, result_dr, csvfile)  # 将结果写入CSV文件
                                csvfile.flush()  # 强制刷新文件缓冲区
                                if result_dr['status'] == 'timeout':  # 如果任何一次运行超时，则标记跳过该算法
                                    dr_timeout = True
                                    print(f"dr 算法 在域大小 {domain_size} 时超时，跳过剩余的域大小")

                            if not ganak_timeout:
                                result_ganak = cnf_ganak(file, domain_size)
                                print(result_ganak)
                                MyTools.write_result(writer, result_ganak, csvfile)  # 将结果写入CSV文件
                                csvfile.flush()  # 强制刷新文件缓冲区
                                if result_ganak['status'] == 'timeout':  # 如果任何一次运行超时，则标记跳过该算法
                                    ganak_timeout = True
                                    print(f"ganak 在域大小 {domain_size} 时超时，跳过剩余的域大小")

                            if not approxmc_timeout:
                                result_approxmc = cnf_approxmc(file, domain_size)

                                # 获取同一domain_size的DR结果进行比较
                                if result_dr['status'] != 'timeout':
                                    dr_result_value = result_dr['result']
                                    approxmc_result_value = result_approxmc['result']

                                    # 确保两个结果都是数值
                                    try:
                                        dr_val = float(dr_result_value) if dr_result_value not in ["TIMEOUT", "ERROR"] else None
                                        approxmc_val = float(approxmc_result_value) if approxmc_result_value not in ["TIMEOUT", "ERROR"] else None

                                        # 只有当两个值都有效时才进行比较
                                        if dr_val is not None and approxmc_val is not None and dr_val >= 1 and approxmc_val >= 1:
                                            # 计算ApproxMC的置信区间（5%）
                                            epsilon = 0.05  # 5%误差范围
                                            lower_bound = approxmc_val * (1 - epsilon)
                                            upper_bound = approxmc_val * (1 + epsilon)

                                            # 检查DR值是否在ApproxMC的置信区间内
                                            within_confidence = (dr_val >= lower_bound) and (dr_val <= upper_bound)

                                            # 添加比较结果到approxmc结果中
                                            result_approxmc['dr_within_approx_confidence'] = within_confidence

                                            # 打印比较结果
                                            within_text = "是" if within_confidence else "否"
                                            print(f"域大小 {domain_size}: DR值={dr_val:.2f}, "
                                                  f"ApproxMC值={approxmc_val:.2f}, "
                                                  f"置信区间=[{lower_bound:.2f}, {upper_bound:.2f}], "
                                                  f"是否在区间内: {within_text}")
                                        else:
                                            result_approxmc['dr_within_approx_confidence'] = None
                                    except (ValueError, TypeError):
                                        result_approxmc['dr_within_approx_confidence'] = None
                                else:
                                    result_approxmc['dr_within_approx_confidence'] = None

                                print(result_approxmc)
                                MyTools.write_result(writer, result_approxmc, csvfile)  # 将结果写入CSV文件
                                csvfile.flush()  # 强制刷新文件缓冲区
                                if result_approxmc['status'] == 'timeout':  # 如果任何一次运行超时，则标记跳过该算法
                                    approxmc_timeout = True
                                    print(f"approxmc 在域大小 {domain_size} 时超时，跳过剩余的域大小")

                            pbar.update(1)  # 更新总体进度条
                MyTools.print_result(result_file_path)


if __name__ == "__main__":
    ## 配置参数
    logger.setLevel(logging.CRITICAL)
    log_path = os.path.join(ExperimentConfig.RESULTS_PATH, "performance.log")
    logfile(log_path, maxBytes=1e6, backupCount=3) # 增加了一个文件日志处理器，并没有移除默认的控制台处理器。
    # logzero.setup_default_logger(console_log_level=None)
    logger.handlers.clear()  # 清除默认 console handler

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取当前时间戳
    # 在结果路径中添加时间戳
    RESULTS_PATH = os.path.join(DIR_PATH, f"results_{timestamp}")  # 存储实验结果的路径
    os.makedirs(RESULTS_PATH, exist_ok=True)  # 创建结果目录

    epsilon = 0.05  # 近似算法的误差范围
    delta = 0.1  # 近似算法的置信度

    ExperimentConfig.RESULTS_PATH = RESULTS_PATH
    ExperimentConfig.TIMEOUT_SECONDS = 50000
    ExperimentConfig.EXPERIMENT_GROUPS = [
        {
            "name": "regular-graphs",
            "domain_sizes": list(range(2, 15, 1)),
            # "domain_sizes": list(range(2, 3, 1)),
            "algorithms": ["dr"],
            "models": {
                "2-regular-graph": "2-regular-graph.wfomcs",
                "3-regular-graph": "3-regular-graph.wfomcs",
                "4-regular-graph": "4-regular-graph.wfomcs",
                # "5-regular-graph": "5-regular-graph.wfomcs",
            }
        },
        # {
        #     "name": "directed-graphs",ls

        #     "domain_sizes": list(range(2, 10, 1)),
        #     # "domain_sizes": list(range(2, 3, 1)),
        #     "algorithms": ["dr"],
        #     "models": {
        #         "2-regular-directed-graph": "2-regular-directed-graph.wfomcs",
        #         # "3-regular-directed-graph": "3-regular-directed-graph.wfomcs",
        #     }
        # },
        # {
        #     "name": "colored-graphs",
        #     "domain_sizes": list(range(2, 10, 1)),
        #     # "domain_sizes": list(range(2, 3, 1)),
        #     "algorithms": ["dr"],
        #     "models": {
        #         "3-regular-graph-2-colored": "3-regular-graph-2-colored.wfomcs",
        #         # "3-regular-graph-3-colored": "3-regular-graph-3-colored.wfomcs",
        #         # "4-regular-graph-2-colored": "4-regular-graph-2-colored.wfomcs",
        #     }
        # }
    ]

    # run_experiment()
    MyTools.print_result("/home/sunshixin/pycharm_workspace/performance/check/results/regular-graphs/baseline_2-regular-graph.csv")
    # MyTools.print_result("/home/sunshixin/pycharm_workspace/experiment2check/WFOMC/check/results_2025-07-29_09-53-04/regular-graphs/baseline_3-regular-graph_20250729_102637.csv")
    # MyTools.print_result("/home/sunshixin/pycharm_workspace/experiment2check/WFOMC/check/results_2025-07-29_09-53-04/regular-graphs/baseline_4-regular-graph_20250729_110035.csv")
