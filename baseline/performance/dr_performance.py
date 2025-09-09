import logging
import csv
from datetime import datetime
from typing import Literal
from multiprocessing import Process, Queue
from zoneinfo import ZoneInfo
from tqdm import tqdm
from wfomc import wfomc, parse_input, Const, Algo
import pandas as pd
import matplotlib.pyplot as plt
import psutil, os, time, threading
import multiprocessing as mp
# 有的环境装了 symengine，有的没有——这里尽量宽容
try:
    from symengine import Number as SymNumber
except Exception:
    SymNumber = None


def monitor_memory(pid: int,
                   interval: float,
                   peak_mem: mp.Value,
                   stop_evt: threading.Event) -> None:
    """
    轮询子进程内存，记录最大 RSS。
    退出条件：子进程结束  或  主线程设置 stop_evt
    """
    try:
        proc = psutil.Process(pid)
        while proc.is_running() and not stop_evt.is_set():
            # 先拿父进程的 USS，再把所有（递归）子进程的 USS 加总
            try:
                rss = proc.memory_full_info().uss  # 只算 Unique Set Size
            except psutil.AccessDenied:
                rss = proc.memory_info().rss  # 回退到 RSS
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_full_info().uss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            with peak_mem.get_lock():  # 原子更新
                if rss > peak_mem.value:
                    peak_mem.value = rss
            stop_evt.wait(interval)  # 代替 time.sleep
    except psutil.NoSuchProcess:
        pass


def print_result(result_data):
    """打印实验结果"""
    result_str = f"""
    {'-' * 20} START {'-' * 20}
    代码运行的时间戳: {datetime.now(ZoneInfo("Asia/Shanghai")).strftime('%Y-%m-%d %H:%M:%S')}
    设置的超时时间: {Config.TIMEOUT_SECONDS} 秒
    公式: {result_data['formula']}
    域大小: {result_data['domain_size']}
    算法: {result_data['algorithm']}
    结果: {result_data['result']}
    耗时: {result_data['time_sec']} 秒
    内存变化: {result_data['memory_bytes']} bytes ({result_data['memory_kb']} KB, {result_data['memory_mb']} MB)
    状态: {result_data['status']}
    {'-' * 20} END {'-' * 20}
    """

    # 打印到控制台
    print(result_str)


def plot_comparison(csv_file_path: str, metric: str, ylabel: str, output_suffix: str, run_time: int = 1) -> None:
    """
    通用的绘图函数，用于绘制算法性能对比图

    参数:
    metric: 要绘制的指标 ('time_sec' 或 'memory_mb')
    ylabel: y轴标签
    output_suffix: 输出文件后缀
    """
    # csv_file = os.path.join(Config.RESULTS_PATH, csv_file_name)

    # 读取CSV文件
    data = pd.read_csv(csv_file_path)
    # 过滤掉状态为 "timeout" 的数据
    if (not os.path.exists(csv_file_path)) or os.path.getsize(csv_file_path) == 0:
        print(f"[skip] CSV 为空，跳过绘图: {csv_file_path}")
        return
    try:
        data = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        print(f"[skip] CSV 无内容可解析，跳过绘图: {csv_file_path}")
        return
    data = data[~data['status'].isin(["timeout", "error"])]

    # 提取数据
    algorithms = data['algorithm'].unique()
    domain_sizes = sorted(data['domain_size'].unique())
    # 创建图形
    plt.figure(figsize=(12, 8))

    # 定义算法在图例中的显示名称
    legend_labels = {
        "dr": "Ours",
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
                 markersize=12,
                 linewidth=3,
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
    pdf_path = os.path.join(result_dir, f"{base_name}_{output_suffix}.pdf")
    plt.savefig(pdf_path, dpi=600, bbox_inches='tight')
    png_path = os.path.join(result_dir, f"{base_name}_{output_suffix}.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    print(f"图表已保存到: {png_path}")
    # plt.show()
    plt.close()


def plot_metric(csv_file_path: str,
                metric: Literal["time_sec", "memory_mb"] = "time_sec",
                run_time: int = 1) -> None:
    mapping = {
        "time_sec": ("Runtime (s)", "time_comparison"),
        "memory_mb": ("Memory Usage (MB)", "memory_comparison"),
    }
    ylabel, suffix = mapping[metric]
    plot_comparison(csv_file_path, metric, ylabel, suffix, run_time)


def plot_kr_comparison(result_file_path: str):
    pass


def run_single(model_name, model_csv_name, domain_size, algo, r=None, k=None):
    """运行单个实验（多进程+超时强制终止）"""
    ## 准备问题
    file_path = os.path.join(Config.MODELS_PATH, model_csv_name)  # 构造模型文件的完整路径
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
    peak_mem = mp.Value('Q', 0)  # 无符号 long long
    stop_evt = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(p.pid, 0.05, peak_mem, stop_evt),
        daemon=True  # 避免极端情况下阻塞主进程退出
    )
    monitor_thread.start()

    p.join(Config.TIMEOUT_SECONDS)
    if p.is_alive():
        p.terminate()
        p.join()
        result = "timeout"
    else:
        result = q.get() if not q.empty() else "error"

    monitor_thread.join()
    end_time = time.time()

    stop_evt.set()  # 通知监控线程退出
    monitor_thread.join(timeout=1)  # 最多等待 1s，防止卡主
    memory_used = peak_mem.value

    return {
        'timestamp': datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S"),  # 当前时间戳
        "formula": model_name,
        "domain_size": domain_size,
        "algorithm": algo,
        "result": result,
        "time_sec": round(end_time - start_time, 4),
        "memory_bytes": memory_used,
        "memory_kb": memory_used / 1024,
        "memory_mb": memory_used / (1024 * 1024),
        "status": "completed" if isinstance(result, SymNumber) else result
    }


def run_experiment(FLUSH_EVERY: int = 20, run_time: int = 1):
    """运行完整的实验"""
    logging.disable(logging.CRITICAL)  # 禁用所有日志输出，只显示CRITICAL级别以上的日志

    # 定义CSV文件的列名 ## 有没有k和r
    fieldnames = ['timestamp', "formula", "domain_size", "algorithm", "result", "time_sec", 'memory_bytes', 'memory_kb', 'memory_mb', 'status']

    ## 获取总的迭代次数
    total_iterations = 0
    for group in Config.GROUPS:
        domain_size = group["domain_size"]
        algorithms = group["algorithms"]
        # 对于每个k值，r的取值范围是0到k-1（共k个值）
        group_iterations = len(domain_size) * len(algorithms) * len(group["models"])
        total_iterations += group_iterations

    with tqdm(total=total_iterations, desc="Overall Progress") as pbar:  # 创建一个总体进度条，显示实验整体进度
        ## 遍历所有实验组
        for group in Config.GROUPS:
            group_name = group["name"]
            domain_size = group["domain_size"]
            algorithms = group["algorithms"]
            models = group["models"]
            sub_dir_path = os.path.join(Config.RESULTS_PATH, group_name)  # 构造子目录的完整路径
            os.makedirs(sub_dir_path, exist_ok=True)

            ## 1 遍历所有模型（公式名称和文件名）
            for model_name, model_csv_name in models.items():
                # result_file_path, result_file_name = generate_results_filename(group_name, model_name)  # 为当前模型生成结果文件名
                subsub_dir_path = os.path.join(sub_dir_path, model_name)  # 为每个模型创建单独的子目录
                os.makedirs(subsub_dir_path, exist_ok=True)
                csv_name = f"{model_name}.csv"  # 为当前模型生成结果文件名
                csv_path = os.path.join(subsub_dir_path, csv_name)  # 为当前模型生成结果文件路径

                with open(csv_path, 'w', newline='') as csvfile:  # 打开结果文件准备写入
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  # 创建CSV字典写入器
                    writer.writeheader()  # 写入CSV文件头部（列名）
                    ## 2 遍历所有需要测试的算法
                    for algo in [a for a in Algo if str(a) in algorithms]:
                        skip_domain_size = False  # 标记是否跳过该算法（当算法超时时）
                        for n in domain_size:  ## 遍历所有域大小
                            if skip_domain_size:  # 如果该算法已超时，则跳过剩余的domain_size
                                pbar.update(1)  # 更新总体进度条
                                continue

                            ## 运行算法
                            single_result = run_single(model_name, model_csv_name, n, algo)  # 运行单个实验，读取model文件，运行wfomc，返回结果
                            # print(f"算法在域大小 {n} 的运行结果: {single_result}")

                            if single_result['status'] == 'timeout':  # 如果任何一次运行超时，则标记跳过该算法
                                skip_domain_size = True
                                print(f"算法在域大小 {n} 时超时，跳过剩余的域大小")

                            result_data = single_result  # 保存最后一次运行结果
                            ## 保存结果到文件
                            if result_data:  # 如果有结果数据则写入文件
                                writer.writerow(result_data)  # 将结果数据写入CSV文件的一行
                                pbar.update(1)  # 更新总体进度条
                                if pbar.n % FLUSH_EVERY == 0:
                                    csvfile.flush()  # 定期强制刷新文件缓冲区
                                print_result(result_data)  # 打印当前结果

                ## 绘图
                for m in ("time_sec", "memory_mb"):  # 为当前模型的结果文件生成运行时间对比图 # 为当前模型的结果文件生成内存使用对比图
                    plot_metric(csv_path, metric=m, run_time=run_time)
                print(f"\n实验完成! 结果保存在: {csv_path}")  # 打印实验完成信息


class Config:
    """实验配置类"""
    # 路径配置
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    MODELS_PATH = os.path.join(DIR_PATH, "models")  # 输入例子的路径
    # 为每次执行创建带时间戳的结果目录
    RESULTS_PATH = os.path.join(DIR_PATH, f"results_{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')}")  # 存储实验结果的路径的文件夹路径，但不是最终csv结果的路径,代码里面又根据不同的model新建了子文件夹
    # 确保结果目录存在
    os.makedirs(RESULTS_PATH, exist_ok=True)
    K_VALUES = range(2, 20)
    # K_VALUES = range(2, 10)
    # MODELS = {
    #     "3-regular-graph-sc2": "3-regular-graph-sc2.wfomcs",
    # }
    ALGORITHMS = ["dr"]
    TIMEOUT_SECONDS = 15000
    GROUPS = [
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
        # {
        #     "name": "directed-graphs",
        #     "domain_size": list(range(2, 25, 1)),
        #     # "domain_size": list(range(2, 5, 3)),
        #     "algorithms": ["dr", "fast"],
        #     "models": {
        #         "2-regular-directed-graph": "2-regular-directed-graph.wfomcs",
        #         "3-regular-directed-graph": "3-regular-directed-graph.wfomcs",
        #     }
        # },
        # {
        #     "name": "colored-graphs",
        #     "domain_size": list(range(2, 45, 1)),
        #     # "domain_size": list(range(2, 5, 3)),
        #     "algorithms": ["dr", "fast"],
        #     "models": {
        #         # "3-regular-graph-2-colored": "3-regular-graph-2-colored.wfomcs",
        #         # "3-regular-graph-3-colored": "3-regular-graph-3-colored.wfomcs",
        #         "3-regular-graph-4-colored": "3-regular-graph-4-colored.wfomcs",
        #         # "4-regular-graph-2-colored": "4-regular-graph-2-colored.wfomcs",
        #         "4-regular-graph-3-colored": "4-regular-graph-3-colored.wfomcs",
        #         "4-regular-graph-4-colored": "4-regular-graph-4-colored.wfomcs"
        #
        #     }
        # }
        {
            "name": "BA",
            "domain_size": list(range(4, 20, 1)),
            "algorithms": ["dr", "fast"],
            "models": {
                "BA": "BA.wfomcs",
            }
        }
    ]


if __name__ == "__main__":
    run_time = 1  # 设置为1表示每个实验只运行一次
    FLUSH_EVERY = 20  # 每 20 条记录刷新一次
    run_experiment(FLUSH_EVERY, run_time)  # 执行实验
