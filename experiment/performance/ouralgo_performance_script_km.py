# 这个脚本引入了Julia
import logging
import csv
from datetime import datetime
from typing import Literal
from multiprocessing import Process, Queue
from zoneinfo import ZoneInfo
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import psutil
import os
import time
import threading
import multiprocessing as mp



class Config:
    """实验配置类，在这里配置参数，比如运行的例子"""

    DIR_PATH = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    
    # --- 只声明变量，不在这里创建文件夹 ---
    RESULTS_PATH = None 
    MODELS_PATH = os.path.join(DIR_PATH, "models")  # 输入例子的路径
    # 为每次执行创建带时间戳的结果目录
    # RESULTS_PATH = os.path.join(
    #     DIR_PATH,
    #     f"results_{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')}",
    # )  # 存储实验结果的路径的文件夹路径，但不是最终csv结果的路径,代码里面又根据不同的model新建了子文件夹
    # os.makedirs(RESULTS_PATH, exist_ok=True)  # 确保结果目录存在
    # K_VALUES = range(2, 20)  # 预留参数，未用
    # K_VALUES = range(2, 10)
    # MODELS = {
    #     "3-regular-graph-sc2": "3-regular-graph-sc2.wfomcs",
    # }
    ALGORITHMS = ["dr"]  # 预留参数，未用
    TIMEOUT_SECONDS = 20000  # 单个实验超时时间（秒）
    # TIMEOUT_SECONDS = 100  # 单个实验超时时间（秒）
    GROUPS = [ # 下面是Julia
        {
            "name": "m-odd-degree-graph-sc2-heatmap",
            "domain_size": [10,],  # 域大小的遍历范围
            "k_values": list(range(45)), # k 的遍历范围
            "m_values": [4,], # m 的遍历范围
            "algorithms": ["dr"], #, "fast"], # 如果 fast 也需要测试，在这里加入
            "models": {
                "m-odd-degree-graph-sc2": "m-odd-degree-graph-sc2.wfomcs",
            },
        },
    ]



def monitor_memory(
    pid: int, interval: float, peak_mem: mp.Value, stop_evt: threading.Event
) -> None:
    """
    在一个单独的线程中轮询指定进程的内存使用情况，并记录峰值。
    退出条件：目标进程结束，或主线程通过stop_evt发出停止信号。
    :param pid: 要监控的子进程的ID。
    :param interval: 轮询的时间间隔（秒）。
    :param peak_mem: 一个多进程共享的Value对象，用于存储峰值内存。
    :param stop_evt: 一个线程事件，用于从外部停止监控。
    """
    try:
        proc = psutil.Process(pid)  # 根据进程ID获取psutil的Process对象
        while (
            proc.is_running() and not stop_evt.is_set()
        ):  # 当进程仍在运行且停止事件未被设置时，持续循环
            # 先拿父进程的 USS，再把所有（递归）子进程的 USS 加总
            try:
                rss = (
                    proc.memory_full_info().uss
                )  # 如果权限不足，则回退到获取RSS（Resident Set Size）
            except psutil.AccessDenied:
                rss = proc.memory_info().rss  # 回退到 RSS
            for child in proc.children(
                recursive=True
            ):  # 遍历所有子进程（递归），将它们的内存使用也加进来
                try:
                    rss += child.memory_full_info().uss  # 累加子进程的USS
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue  # 如果子进程已消失或无法访问，则跳过
            with peak_mem.get_lock():  # 使用锁来保证对共享内存的原子性更新
                if rss > peak_mem.value:  # 如果当前内存使用超过记录的峰值
                    peak_mem.value = rss  # 更新峰值
            stop_evt.wait(
                interval
            )  # 等待指定的时间间隔，这种方式可以被stop_evt.set()立即中断
    except psutil.NoSuchProcess:
        pass  # 如果进程在监控开始前就结束了，则忽略异常并退出


def print_result(result_data):
    """打印实验结果"""
    result_str = f"""
    {'-' * 20} START {'-' * 20}
    代码运行的时间戳: {datetime.now(ZoneInfo("Asia/Shanghai")).strftime('%Y-%m-%d %H:%M:%S')}
    设置的超时时间: {Config.TIMEOUT_SECONDS} 秒
    公式: {result_data['formula']}
    k值: {result_data['k_value']}
    m值: {result_data['m_value']}
    算法: {result_data['algorithm']}
    结果: {result_data['result']}
    耗时: {result_data['time_sec']} 秒
    内存变化: {result_data['memory_bytes']} bytes ({result_data['memory_kb']} KB, {result_data['memory_mb']} MB)
    状态: {result_data['status']}
    {'-' * 20} END {'-' * 20}
    """

    # 打印到控制台
    print(result_str)


def plot_comparison(
    csv_file_path: str,
    metric: str = "time_sec",
    ylabel: str = "Runtime (s)",
    output_suffix: str = "time_comparison",
    repeated_run_time: int = 1,
) -> None:
    """
    一个通用的绘图函数，根据给定的CSV文件和指标（时间或内存），绘制算法性能对比图。
    :param csv_file_path: 结果CSV文件的路径。
    :param metric: 要绘制的指标，如 'time_sec' 或 'memory_mb'。
    :param ylabel: 图表Y轴的标签。
    :param output_suffix: 输出图片文件名的后缀。
    :param repeated_run_time: 实验运行次数，用于判断是取单次值还是平均值。
    """
    # csv_file = os.path.join(Config.RESULTS_PATH, csv_file_name)
    
    # 检查CSV文件是否存在或为空，如果是则跳过绘图
    if (not os.path.exists(csv_file_path)) or os.path.getsize(
        csv_file_path
    ) == 0:  
        print(f"[skip] CSV 为空，跳过绘图: {csv_file_path}")
        return
    try:
        data = pd.read_csv(csv_file_path)  # 使用pandas读取CSV文件
    except pd.errors.EmptyDataError:  # 如果CSV文件只有头部没有内容，pandas会报错
        print(f"[skip] CSV 无内容可解析，跳过绘图: {csv_file_path}")
        return
    # 过滤掉状态为 "timeout" 或 "error" 的数据行，这些数据不参与绘图
    data = data[
        ~data["status"].isin(["timeout", "error"])
    ]  

    # 提取数据
    algorithms = data["algorithm"].unique()  # 获取所有不重复的算法名称
    domain_sizes = sorted(data["domain_size"].unique())  # 获取所有不重复的域大小并排序
    
    # 创建一个新的图表，设置尺寸
    plt.figure(figsize=(12, 8))
    # 定义图例中各个算法的显示名称
    legend_labels = {"dr": "Ours", "fast": "Fast", "incremental": "Incremental"}
    # 定义不同算法在图上的标记样式
    markers = {
        "dr": "o",  # 圆形标记
        "fast": "s",  # 正方形标记
        "incremental": "s",  # 三角形标记
    }
    # 遍历每一种算法，分别绘制它们的性能曲线
    for algo in algorithms:
        algo_data = data[data["algorithm"] == algo]  # 筛选出当前算法的数据
        if repeated_run_time == 1:  # 如果每个实验只运行一次，因为有可能多次运行取平均值，但是我没有使用，所以一直是1
            values = [  # 直接提取每个域大小对应的指标值
                (
                    algo_data[algo_data["domain_size"] == size][metric].iloc[0]
                    if not algo_data[algo_data["domain_size"] == size].empty
                    else float("nan")
                )
                for size in domain_sizes
            ]
        else:  # 如果运行多次，则计算平均值
            values = [
                algo_data[algo_data["domain_size"] == size][metric].mean()
                for size in domain_sizes
            ]

        plt.plot(  # 绘制当前算法的性能曲线
            domain_sizes,
            values,
            marker=markers.get(algo, "o"),
            label=legend_labels.get(algo, algo),
            markersize=12,
            linewidth=3,
            markeredgecolor="black",
            markeredgewidth=1,
        )
        
    # 如果绘制的是时间，Y轴使用对数刻度
    if metric == "time_sec":  
        plt.yscale("log")

    ax = plt.gca() # 这里采用获取绘图对象的方式，是未来设置刻度轴为整数
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # 设置X轴刻度为整数

    plt.xlabel("Domain Size", fontsize=18)  # 设置X轴标签和字体大小
    plt.ylabel(ylabel, fontsize=18)  # 设置Y轴标签和字体大小
    # plt.legend(fontsize=18, loc='upper left')
    plt.legend(fontsize=18)  # 显示图例
    plt.xticks(fontsize=18)  # 设置X轴刻度字体大小
    plt.yticks(fontsize=18)  # 设置Y轴刻度字体大小
    plt.grid(True, linestyle="--", alpha=0.7)  # 显示网格线

    ## 保存图片到本地
    # 去除文件名后缀并生成输出路径
    # csv_file_path = Path(csv_file_path)  # 使用Path对象处理文件路径
    file_name = os.path.basename(csv_file_path)  # 获取CSV文件名
    base_name = os.path.splitext(file_name)[0]  # 去掉文件扩展名
    result_dir = os.path.dirname(csv_file_path)  # 获取文件所在目录
    pdf_path = os.path.join(
        result_dir, f"{base_name}_{output_suffix}.pdf"
    )  # PDF图片路径
    plt.savefig(pdf_path, dpi=600, bbox_inches="tight")  # 保存为PDF格式
    png_path = os.path.join(
        result_dir, f"{base_name}_{output_suffix}.png"
    )  # PNG图片路径
    plt.savefig(png_path, dpi=600, bbox_inches="tight")  # 保存为PNG格式
    print(f"图表已保存到: {png_path}")
    # plt.show()
    plt.close()  # 关闭当前图表，释放内存


def plot_metric(
    csv_file_path: str,
    metric: Literal["time_sec", "memory_mb"] = "time_sec",
    repeated_run_time: int = 1,
) -> None:
    """一个便捷函数，根据指定的metric调用通用的绘图函数plot_comparison"""
    mapping = {  # 定义不同metric对应的Y轴标签和文件名后缀
        "time_sec": ("Runtime (s)", "time_comparison"),
        "memory_mb": ("Memory Usage (MB)", "memory_comparison"),
    }
    ylabel, suffix = mapping[metric]
    plot_comparison(csv_file_path, metric, ylabel, suffix, repeated_run_time)





def python_worker(q, model_name, model_csv_name, domain_size, k_value, m_value, algo):
    """这个函数在独立的子进程中运行，负责调用 Python 的 wfomc。"""
    try:
        from wfomc import wfomc, parse_input, Const, Algo
        # 将 problem 的构建逻辑移到这里
        file_path = os.path.join(Config.MODELS_PATH, model_csv_name)
                
        # 这段代码用了测试odd 中固定domain, 但是odd 里面的 m 是变量，这里用domain_size 替换掉 m

        # 2. 如果是特定模型，则进行字符串替换
        if model_name == "m-odd-degree-graph-sc2":
            final_model_string = r"""
            \forall X: (~E(X,X)) &
            \forall X: (\forall Y: (E(X,Y) -> E(Y,X))) &
            \forall X: (P(X) <-> (~Odd(X) & A(X) & C(X))) &
            \forall X: (\forall Y: (P(X) & B(X,Y) -> U(Y))) &
            \forall X: (\forall Y:(~P(X) -> (B(X,Y) <-> E(X,Y)))) &
            \forall X: (Odd(X) | A(X)) &
            \forall X: (A(X) | C(X)) &
            \forall X: (\exists_{1 mod 2} Y: (B(X, Y))) &
            \exists_{=m} X: (Odd(X)) &
            \exists_{=1} X: (U(X)) 

            n = 3
            1 -1 C
            |E| = k
            """
            # 使用 domain_size 变量的值替换模板中的 {m} 占位符
            # 使用 re.sub 替换 m 和 k 的值
            # 匹配 \exists_{=m} 并替换为 \exists_{=<m_value>}
            import re
            final_model_string = re.sub(r'\\exists_{=m}', fr'\\exists_{{={m_value}}}', final_model_string)
            # 匹配 |E| = k 并替换为 |E| = <k_value>
            final_model_string = re.sub(r'\|E\| = k', fr'|E| = {k_value}', final_model_string)
            
        # 3. 使用 wfomcs_parse API 解析最终构建好的字符串
        from wfomc.parser.wfomcs_parser import parse as wfomcs_parse
        problem = wfomcs_parse(final_model_string)
        
        problem.domain = {Const(f"d{i}") for i in range(domain_size)} # --------------注意绝大部分例子是需要这句话，但是m odd,为了输入变量为m ,需要注释
        
        # 这里是为了测试m-odd degree, 因为 |E| = 2n。也就是(总边数，随 n 动态变化)。确保了图随着 n 的增长而不会变得过于稀疏或稠密。
        # if model_name == "m-odd-degree-graph-sc2":
        #     if (
        #         problem.cardinality_constraint
        #         and problem.cardinality_constraint.constraints
        #     ):
        #         old_E_constraint = (
        #             problem.cardinality_constraint.constraints[0]
        #         )  # 假设第一个约束是关于 E 的
        #         new_E_constraint = (
        #             old_E_constraint[0],
        #             old_E_constraint[1],
        #             2 * domain_size,
        #         )  # 假设约束的第三个参数（通常是数量）与域大小相关
        #         problem.cardinality_constraint.constraints[0] = new_E_constraint
        #         # 这里是为了测试BA，因为|Eq|=n, 所以才添加的代码。这段代码只针对这个特殊例子。
                

        # 调用核心计算函数
        res = wfomc(problem, algo)
        q.put(res)
    except Exception as e:
        q.put(f"python_error: {e}")

# 这里是不包含Julia的原始版本
def run_python_single(model_name, model_csv_name, n, m, k, algo):
    """
    运行单个实验配置。使用子进程来执行，以便于设置超时和监控资源。
    :return: 一个包含实验结果的字典。
    """
    start_time = time.time()
    q = Queue()
    # target 现在是顶层的 python_worker 函数 ---
    p = Process(target=python_worker, args=(q, model_name, model_csv_name, n, k, m, algo))
    p.start()
    # 创建并启动内存监控线程
    peak_mem = mp.Value(
        "Q", 0
    )  # 创建一个共享内存值，用于存储内存峰值（'Q'表示无符号长整型）
    stop_evt = threading.Event()  # 创建一个事件，用于通知监控线程停止
    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(
            p.pid,
            0.05,
            peak_mem,
            stop_evt,
        ),  # 传入子进程ID、间隔、共享内存和停止事件
        daemon=True,  # 设置为守护线程，这样主进程退出时它也会退出
    )
    monitor_thread.start()

    p.join(Config.TIMEOUT_SECONDS)  # 等待子进程结束，但最多等待指定的超时时间
    if p.is_alive():  # 如果等待超时后子进程仍在运行
        p.terminate()  # 强制终止子进程
        p.join()  # 等待终止完成
        result = "timeout"  # 结果标记为超时
    else:  # 如果子进程正常结束
        result = (
            q.get() if not q.empty() else "error"
        )  # 从队列中获取结果，如果队列为空则标记为错误

    monitor_thread.join()
    end_time = time.time()  # 记录结束时间

    stop_evt.set()  # 设置停止事件，通知内存监控线程退出
    monitor_thread.join(timeout=1)  # 等待监控线程结束，最多等待1秒
    memory_used = peak_mem.value  # 从共享内存中读取记录的内存峰值
    from symengine import Number as SymNumber

    return {  # 返回一个包含所有结果信息的字典
        "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
            "%Y%m%d_%H%M%S"
        ),  # 当前时间戳
        "formula": model_name,
        "domain_size": n,
        "k_value": k,
        "m_value": m,
        "algorithm": algo,
        "result": result,
        "time_sec": round(end_time - start_time, 4),
        "memory_bytes": memory_used,
        "memory_kb": memory_used / 1024,
        "memory_mb": memory_used / (1024 * 1024),
        "status": "completed" if isinstance(result, SymNumber) else result,
    }


def run_experiment(FLUSH_EVERY: int = 20, repeated_run_time: int = 1):
    from wfomc import Algo
    """运行完整的实验"""
    logging.disable(logging.CRITICAL)  # 禁用所有日志输出，只显示CRITICAL级别以上的日志

    # 定义CSV文件的列名 ## 有没有k和r
    fieldnames = [
        "timestamp",
        "formula",
        "domain_size",
        "k_value",
        "m_value",
        "algorithm",
        "result",
        "time_sec",
        "memory_bytes",
        "memory_kb",
        "memory_mb",
        "status",
    ]

    # 计算总的实验迭代次数，用于tqdm进度条
    total_iterations = 0
    for group in Config.GROUPS:
        domain_size = group["domain_size"]
        algorithms = group["algorithms"]
        # 对于每个k值，r的取值范围是0到k-1（共k个值）
        group_iterations = len(domain_size) * len(algorithms) * len(group["models"])
        total_iterations += group_iterations

    # 创建一个总进度条
    with tqdm(
        total=total_iterations, desc="Overall Progress"
    ) as pbar:  # 创建一个总体进度条，显示实验整体进度
        # 遍历所有实验组
        for group in Config.GROUPS:
            group_name = group["name"]
            domain_size = group["domain_size"]
            algorithms = group["algorithms"]
            models = group["models"]
            sub_dir_path = os.path.join(  # 构造当前实验组的结果子目录
                Config.RESULTS_PATH, group_name
            )  # 构造子目录的完整路径
            os.makedirs(sub_dir_path, exist_ok=True)

            ## 1 遍历所有模型（公式名称和文件名）
            for model_name, model_csv_name in models.items():
                # result_file_path, result_file_name = generate_results_filename(group_name, model_name)  # 为当前模型生成结果文件名
                subsub_dir_path = os.path.join(
                    sub_dir_path, model_name
                )  # 为每个模型创建单独的子目录
                os.makedirs(subsub_dir_path, exist_ok=True)
                csv_name = f"{model_name}.csv"  # 结果CSV文件名和路径
                csv_path = os.path.join(
                    subsub_dir_path, csv_name
                )  # 为当前模型生成结果文件路径

                with open(csv_path, "w", newline="") as csvfile:  # 打开结果文件准备写入
                    writer = csv.DictWriter(
                        csvfile, fieldnames=fieldnames
                    )  # 创建CSV字典写入器
                    writer.writeheader()  # 写入CSV文件头部（列名）
                    ## 2 遍历所有需要测试的算法
                    for algo in [a for a in Algo if str(a) in algorithms]:
                        k_values = group.get("k_values", [])
                        m_values = group.get("m_values", [])
                        domain_size = group["domain_size"]
                        
                        for n in domain_size:
                            for k in k_values:
                                skip = False  # 标记是否跳过该算法（当算法超时时）
                                for m in m_values:
                                    if (
                                        skip
                                    ):  # 如果该算法已超时，则跳过剩余的domain_size
                                        pbar.update(1)  # 更新总体进度条
                                        continue
                                    
                                    single_result = run_python_single(
                                        model_name, model_csv_name, n, m, k, algo
                                    )  # 运行单个实验，读取model文件，运行wfomc，返回结果
                                    # print(f"算法在域大小 {n} 的运行结果: {single_result}")

                                    if (
                                        single_result["status"] == "timeout" or single_result["status"] == "hard_timeout" or single_result["status"] == "error"
                                    ):  # 如果任何一次运行超时，则标记跳过该算法
                                        skip = True
                                        print(f"算法在域大小 {n} 时超时，跳过剩余的域大小")

                                    result_data = single_result  # 保存最后一次运行结果
                                    # 保存结果到文件
                                    if result_data:  # 如果有结果数据则写入文件
                                        writer.writerow(
                                            result_data
                                        )  # 将结果数据写入CSV文件的一行
                                        pbar.update(1)  # 更新总体进度条
                                        if pbar.n % FLUSH_EVERY == 0:
                                            csvfile.flush()  # 定期强制刷新文件缓冲区
                                        print_result(result_data)  # 打印当前结果

                # 绘图
                for m in (
                    "time_sec",
                    "memory_mb",
                ):  # 为当前模型的结果文件生成运行时间对比图 # 为当前模型的结果文件生成内存使用对比图
                    plot_metric(csv_path, metric=m, repeated_run_time=repeated_run_time)
                print(f"\n实验完成! 结果保存在: {csv_path}")  # 打印实验完成信息


if __name__ == "__main__":
    # 2. 创建唯一的、本次实验的文件夹
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    Config.RESULTS_PATH = os.path.join(Config.DIR_PATH, f"results_{timestamp}")
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    
    print(f"实验结果将保存在: {Config.RESULTS_PATH}")
    
    repeated_run_time = 1  # 设置为1表示每个实验只运行一次  这个参数没有用，在代码中可能没有具体的实现，但是先保留，防止以后需要
    FLUSH_EVERY = 1  # 每 5 条记录刷新一次
    run_experiment(FLUSH_EVERY, repeated_run_time)  # 执行实验
