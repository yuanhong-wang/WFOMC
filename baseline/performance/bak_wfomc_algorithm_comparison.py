import json
import logging
import csv
import os
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue
from contexttimer import Timer
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import psutil
from tqdm import tqdm
import time
from wfomc import wfomc, parse_input, Const, Algo
import pandas as pd
import matplotlib.pyplot as plt
from logzero import logger, logfile
import psutil, os, time, threading


def monitor_memory(pid, interval, peak_mem):
    try:
        p = psutil.Process(pid)
        while p.is_running():
            mem = p.memory_info().rss
            peak_mem[0] = max(peak_mem[0], mem)
            time.sleep(interval)
    except psutil.NoSuchProcess:
        pass



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


def generate_results_filename(subdir, name):
    """生成带时间戳的结果文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间并格式化为字符串
    result_dir = os.path.join(ExperimentConfig.RESULTS_PATH, subdir)
    if os.path.exists(result_dir) is False:
        os.makedirs(result_dir)  # 如果结果目录不存在，则创建它
    file_name = f"baseline_{name}_{timestamp}.csv" # 生成的时间戳
    return os.path.join(result_dir, file_name), file_name


def write_result(writer, result_data, csvfile):
    """即时写入并落盘"""
    writer.writerow(result_data)  # 将结果数据写入CSV文件的一行
    csvfile.flush()  # 强制将缓冲区的数据写入磁盘
    os.fsync(csvfile.fileno())  # 强制操作系统将数据从内核缓冲区写入磁盘，确保数据持久化


def print_result(result_data):
    """打印实验结果"""
    result_str = "\n" + "-" * 20 + " START " + "-" * 20 + "\n"
    result_str += f"代码运行的时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    result_str += f"公式: {result_data['formula']}\n"
    result_str += f"域大小: {result_data['domain_size']}\n"
    result_str += f"算法: {result_data['algorithm']}\n"
    result_str += f"结果: {result_data['result']}\n"
    result_str += f"耗时: {result_data['time_sec']} 秒\n"
    result_str += f"内存变化: {result_data['memory_bytes']} bytes "
    result_str += f"({result_data['memory_kb']} KB, "
    result_str += f"{result_data['memory_mb']} MB)\n"
    result_str += f"状态: {result_data['status']}\n"
    result_str += "-" * 20 + " END " + "-" * 20

    # 打印到控制台
    print(result_str)
    # 先把所有值转为字符串，保证 JSON 可序列化


def plot_comparison(csv_file_path: str, metric: str, ylabel: str, output_suffix: str):
    """
    通用的绘图函数，用于绘制算法性能对比图

    参数:
    metric: 要绘制的指标 ('time_sec' 或 'memory_mb')
    ylabel: y轴标签
    output_suffix: 输出文件后缀
    """
    # csv_file = os.path.join(ExperimentConfig.RESULTS_PATH, csv_file_name)

    # 读取CSV文件
    data = pd.read_csv(csv_file_path)
    # 过滤掉状态为 "timeout" 的数据
    data = data[data['status'] != "timeout"]

    # 提取数据
    algorithms = data['algorithm'].unique()
    domain_sizes = sorted(data['domain_size'].unique())

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 定义算法在图例中的显示名称
    legend_labels = {
        "dr": "OurAlgo",
        "fast": "Fast",
        "recursive": "Recursive"
    }

    # 定义不同算法的标记样式
    markers = {
        "dr": "o",  # 圆形标记
        "fast": "s",  # 正方形标记
        "recursive": "^"  # 三角形标记

    }

    for algo in algorithms:
        algo_data = data[data['algorithm'] == algo]
        if run_time == 1:
            values = [
                algo_data[algo_data['domain_size'] == size][metric].iloc[0]
                if not algo_data[algo_data['domain_size'] == size].empty else float('nan')
                for size in domain_sizes
            ]
        else:
            values = [algo_data[algo_data['domain_size'] == size][metric].mean() for size in domain_sizes]

        plt.plot(domain_sizes, values, marker=markers.get(algo, "o"),
                 label=legend_labels.get(algo, algo),
                 markersize=5,
                 linewidth=2,
                 markeredgecolor='black',
                 markeredgewidth=1
                 )

    # 设置图表属性
    if metric == 'time_sec':
        plt.yscale('log')

    plt.xlabel('Domain Size', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    # plt.legend(fontsize=18, loc='upper left')
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图片到本地
    # 去除文件名后缀并生成输出路径
    # csv_file_path = Path(csv_file_path)  # 使用Path对象处理文件路径
    file_name = os.path.basename(csv_file_path)
    base_name = os.path.splitext(file_name)[0]
    result_dir = os.path.dirname(csv_file_path)
    output_path = os.path.join(result_dir, f"{base_name}_{output_suffix}.pdf")
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf')
    output_path = os.path.join(result_dir, f"{base_name}_{output_suffix}.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png')
    print(f"图表已保存到: {output_path}")

    # 显示图形
    # plt.show()
    plt.close()

def plot_runtime_comparison(result_file_path: str):
    """
    读取CSV文件并绘制不同算法的运行时间对比图
    """
    plot_comparison(result_file_path, 'time_sec', 'Runtime (s)', 'time_comparison')

def plot_memory_comparison(result_file_path: str):
    """
    读取CSV文件并绘制不同算法的内存使用对比图
    """
    plot_comparison(result_file_path, 'memory_mb', 'Memory Usage (MB)', 'memory_comparison')







def _run_single_experiment(formula_name, file, domain_size, algo):
    """运行单个实验（多进程+超时强制终止）"""
    ## 准备问题
    file_path = os.path.join(ExperimentConfig.MODELS_PATH, file)  # 构造模型文件的完整路径
    problem = parse_input(file_path)  # 解析模型文件，获取问题定义
    problem.domain = {Const(f'd{i}') for i in range(domain_size)}  # 根据指定的域大小构造域集合

    start_time = time.time()
    q = Queue()

    def wrapper(q):
        res = wfomc(problem, algo)
        q.put(res)

    p = Process(target=wrapper, args=(q,))
    p.start()

    # 监控子进程内存
    peak_mem = [0]
    monitor_thread = threading.Thread(target=monitor_memory, args=(p.pid, 0.05, peak_mem))
    monitor_thread.start()

    p.join(ExperimentConfig.TIMEOUT_SECONDS)
    if p.is_alive():
        p.terminate()
        p.join()
        result = "TIMEOUT"
    else:
        result = q.get() if not q.empty() else "ERROR"

    monitor_thread.join()
    end_time = time.time()

    memory_used = peak_mem[0]  # 取子进程运行期间的最大RSS
    return {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),  # 当前时间戳
        "formula": formula_name,
        "domain_size": domain_size,
        "algorithm": algo,
        "result": result,
        "time_sec": round(end_time - start_time, 4),
        "memory_bytes": memory_used,
        "memory_kb": memory_used / 1024,
        "memory_mb": memory_used / (1024 * 1024),
        "status": "completed" if result not in ("TIMEOUT", "ERROR") else "timeout"
    }


def run_experiment(self):
    """运行完整的实验"""
    # 禁用所有日志输出，只显示CRITICAL级别以上的日志
    logging.disable(logging.CRITICAL)

    # 定义CSV文件的列名
    fieldnames = ['timestamp', 'formula', 'domain_size', 'algorithm',
                  'result', 'time_sec', 'memory_bytes', 'memory_kb',
                  'memory_mb', 'status']

    ## 遍历所有实验组
    for group in ExperimentConfig.EXPERIMENT_GROUPS:
        group_name = group["name"]
        domain_sizes = group["domain_sizes"]
        algorithms = group["algorithms"]
        models = group["models"]

        print(f"\n开始执行实验组: {group_name}，域大小范围: {domain_sizes}，算法列表: {algorithms}，模型列表: {list(models.keys())}")

        # 获取总的迭代次数
        total_iterations = len(models) * len(domain_sizes) * len([a for a in Algo if str(a) in algorithms])

        with tqdm(total=total_iterations, desc="Overall Progress") as pbar:  # 创建一个总体进度条，显示实验整体进度
            ## 1 遍历所有模型（公式名称和文件名）
            for formula_name, file in models.items():
                result_file_path, result_file_name = generate_results_filename(group_name, formula_name)  # 为当前模型生成结果文件名
                with open(result_file_path, 'w', newline='') as csvfile:  # 打开结果文件准备写入
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # 创建CSV字典写入器
                    writer.writeheader()  # 写入CSV文件头部（列名）
                    ## 2 遍历所有需要测试的算法
                    for algo in [a for a in Algo if str(a) in algorithms]:
                        skip_domain_size = False  # 标记是否跳过该算法（当算法超时时）
                        for domain_size in domain_sizes:  ## 遍历所有域大小
                            if skip_domain_size:  # 如果该算法已超时，则跳过剩余的domain_size
                                pbar.update(1)  # 更新总体进度条
                                continue

                            ## 运行单个实验
                            result_data = None
                            for _ in range(run_time):  # 根据run_time参数决定运行次数（用于取平均值）
                                single_result = self._run_single_experiment(  # 执行单次实验
                                    formula_name, file, domain_size, algo)

                                # 如果任何一次运行超时，则标记跳过该算法
                                if single_result['status'] == 'timeout':
                                    skip_domain_size = True
                                    print(f"算法 {algo} 在域大小 {domain_size} 时超时，跳过剩余的域大小")

                                # 保存最后一次运行结果
                                result_data = single_result

                                ## 记录结果
                                if result_data:  # 如果有结果数据则写入文件
                                    write_result(writer, result_data, csvfile)  # 将结果写入CSV文件
                                    csvfile.flush()  # 强制刷新文件缓冲区

                                    print_result(result_data)  # 打印当前结果

                                pbar.update(1)  # 更新总体进度条
                ## 绘图
                plot_runtime_comparison(result_file_path)  # 为当前模型的结果文件生成运行时间对比图
                plot_memory_comparison(result_file_path)  # 为当前模型的结果文件生成内存使用对比图
                print(f"\n实验完成! 结果保存在: {result_file_path}")  # 打印实验完成信息


def main():
    log_path = os.path.join(ExperimentConfig.RESULTS_PATH, "performance.log")
    # 使用disableStderrLogger=True参数来禁用控制台输出，只写入文件
    # logfile(log_path, maxBytes=1e6, backupCount=3, disableStderrLogger=True)
    logfile(log_path, maxBytes=1e6, backupCount=3)
    # 设置日志等级（只显示 INFO 及以上）
    logger.setLevel(logging.INFO)

    # 记录实验启动信息
    logger.info("=" * 50)
    logger.info("实验开始")
    logger.info(f"实验配置 - 超时时间: {ExperimentConfig.TIMEOUT_SECONDS}s")
    logger.info(f"结果保存路径: {ExperimentConfig.RESULTS_PATH}")
    total_experiments = sum(
        len(group["models"]) * len(group["domain_sizes"]) * len([a for a in Algo if str(a) in group["algorithms"]])
        for group in ExperimentConfig.EXPERIMENT_GROUPS
    )
    logger.info(f"总计实验次数: {total_experiments}")
    logger.info("=" * 50)

    """主函数"""
    run_experiment()

    # 记录实验结束信息
    logger.info("=" * 50)
    logger.info("实验结束")
    logger.info("=" * 50)


class ExperimentConfig:
    """实验配置类"""
    # 路径配置
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录

    MODELS_PATH = os.path.join(DIR_PATH, "models")  # 输入例子的路径
    # 为每次执行创建带时间戳的结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_PATH = os.path.join(DIR_PATH, f"results_{timestamp}")  # 存储实验结果的路径的文件夹路径，但不是最终csv结果的路径,代码里面又根据不同的model新建了子文件夹
    # 确保结果目录存在
    os.makedirs(RESULTS_PATH, exist_ok=True)
    TIMEOUT_SECONDS = 50000  # 超市时间，5分钟超时

    # 创建结果目录
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # 实验参数，示例，后面可以修改
    DOMAIN_SIZES = list(range(2, 70, 2))
    MODELS = {
        "3-regular-graph-sc2": "3-regular-graph-sc2.wfomcs",
        # "4-regular-graph-sc2": "4-regular-graph-sc2.wfomcs",
        # "5-regular-graph-sc2": "5-regular-graph-sc2.wfomcs",
    }

    # 算法配置，示例，后面可以修改
    ALGORITHMS = ["dr", "recursive", "fast"]

    # 实验组配置：每组包含特定的域大小范围、算法列表和模型列表
    EXPERIMENT_GROUPS = dict()

if __name__ == "__main__":
    # 需要在这里配置参数
    ExperimentConfig.EXPERIMENT_GROUPS = [
        # {
        #     "name": "regular-graphs",
        #     "domain_sizes": list(range(2, 70, 3)),
        #     # "domain_sizes": list(range(2, 7, 3)),
        #     "algorithms": ["dr", "fast"],
        #     "models": {
        #         "3-regular-graph": "3-regular-graph.wfomcs",
        #         "4-regular-graph": "4-regular-graph.wfomcs",
        #         "5-regular-graph": "5-regular-graph.wfomcs",
        #     }
        # },
        {
            "name": "directed-graphs",
            "domain_sizes": list(range(2, 25, 3)),
            # "domain_sizes": list(range(2, 5, 3)),
            "algorithms": ["dr", "fast"],
            "models": {
                "2-regular-directed-graph": "2-regular-directed-graph.wfomcs",
                "3-regular-directed-graph": "3-regular-directed-graph.wfomcs",
            }
        },
        {
            "name": "colored-graphs",
            "domain_sizes": list(range(2, 45, 3)),
            # "domain_sizes": list(range(2, 5, 3)),
            "algorithms": ["dr", "fast"],
            "models": {
                "3-regular-graph-2-colored": "3-regular-graph-2-colored.wfomcs",
                "3-regular-graph-3-colored": "3-regular-graph-3-colored.wfomcs",
                "3-regular-graph-4-colored": "3-regular-graph-4-colored.wfomcs",
                "4-regular-graph-2-colored": "4-regular-graph-2-colored.wfomcs",
                "4-regular-graph-3-colored": "4-regular-graph-3-colored.wfomcs",
                "4-regular-graph-4-colored": "4-regular-graph-4-colored.wfomcs"

            }
        }
    ]

    # 需要先定义run_time变量，因为它在其他函数中被使用
    run_time = 1
    main()
