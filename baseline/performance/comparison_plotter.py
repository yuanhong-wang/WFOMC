import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

run_time = 1


def plot_comparison(csv_file_name: str, metric: str, ylabel: str, output_suffix: str):
    """
    通用的绘图函数，用于绘制算法性能对比图

    参数:
    csv_file_name: CSV文件名
    metric: 要绘制的指标 ('time_sec' 或 'memory_mb')
    ylabel: y轴标签
    output_suffix: 输出文件后缀
    """
    global RESULT_PATH
    csv_file = os.path.join(RESULT_PATH, csv_file_name)

    # 读取CSV文件
    data = pd.read_csv(csv_file)
    # 过滤掉状态为 "timeout" 的数据
    data = data[data['status'] != "timeout"]

    # 提取数据
    algorithms = data['algorithm'].unique()
    domain_sizes = sorted(data['domain_size'].unique())

    # 创建图形
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 18})  # 设置全局字体大小

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
                 markeredgewidth=1)

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
    base_name = os.path.splitext(csv_file_name)[0]
    output_path = os.path.join(RESULT_PATH, f"{base_name}_{output_suffix}.pdf")
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='pdf')
    output_path = os.path.join(RESULT_PATH, f"{base_name}_{output_suffix}.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png')
    print(f"图表已保存到: {output_path}")

    # 显示图形
    plt.show()


def plot_runtime_comparison(csv_file_name: str):
    """
    读取CSV文件并绘制不同算法的运行时间对比图
    """
    plot_comparison(csv_file_name, 'time_sec', 'Runtime (s)', 'time_comparison')


def plot_memory_comparison(csv_file_name: str):
    """
    读取CSV文件并绘制不同算法的内存使用对比图
    """
    plot_comparison(csv_file_name, 'memory_mb', 'Memory Usage (MB)', 'memory_comparison')



if __name__ == '__main__':
    # DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    # print(f"当前目录: {DIR_PATH}")
    RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results","regular-graphs")
    plot_runtime_comparison("baseline_3-regular-graph.csv")
    plot_runtime_comparison("baseline_4-regular-graph.csv")
    plot_runtime_comparison("baseline_5-regular-graph.csv")
    plot_memory_comparison("baseline_3-regular-graph.csv")
    plot_memory_comparison("baseline_4-regular-graph.csv")
    plot_memory_comparison("baseline_5-regular-graph.csv")

    # RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results","colored-graphs")
    # plot_runtime_comparison("3-regular-graph-2-colored.csv")
    # plot_runtime_comparison("3-regular-graph-3-colored.csv")
    # plot_runtime_comparison("3-regular-graph-4-colored.csv")
    # plot_runtime_comparison("4-regular-graph-2-colored.csv")
    # plot_runtime_comparison("4-regular-graph-3-colored.csv")
    # plot_runtime_comparison("4-regular-graph-4-colored.csv")


    # RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results","directed-graphs")
    # plot_runtime_comparison("2-regular-directed-graph.csv")
    # plot_runtime_comparison("3-regular-directed-graph.csv")
    #======================================================


