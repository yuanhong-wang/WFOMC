import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

run_time = 1

plt.rcParams.update({
    'font.family': 'Times New Roman',  # 确保使用常见的字体
    'font.size': 18
})
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
    if not os.path.exists(csv_file):
        print(f"CSV file '{csv_file}' does not exist.")

    # csv_file = csv_file_name

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

    plt.xlabel('Domain Size',fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
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


def print_check_result(RESULT_PATH, csv_file):
    csv_path = os.path.join(RESULT_PATH, csv_file)  # 构造CSV文件的完整路径

    # 1. 读取 CSV
    data = pd.read_csv(csv_path)
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
    # plt.rcParams.update({'font.size': 18})  # 设置全局字体大小

    if not ouralgo_value.empty:
        plt.plot(ouralgo_domain, ouralgo_value, label="OurAlgo", color='#ff7f0e', linewidth=2)
    if not ganak_values.empty:
        plt.scatter(ganak_domain, ganak_values, color='#1f77b4', marker='s', s=50, label="Ganak (Exact)")
    if not approx_values.empty:
        plt.errorbar(approx_domain, approx_values,
                     yerr=[approx_values - approx_lower, approx_upper - approx_values],
                     fmt='o', color='red', ecolor='red', elinewidth=1, capsize=6,markersize=5,label="ApproxMC (95% CI)")

    # 5. 坐标轴 & 样式
    plt.yscale("log")
    plt.xlabel("Domain Size",fontsize=20)
    plt.ylabel("Model Count (log scale)",fontsize=20)
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
    os.makedirs(RESULT_PATH, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    pdf_path = os.path.join(RESULT_PATH, f"{formula}_check.pdf")
    png_path = os.path.join(RESULT_PATH, f"{formula}_check.png")

    plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    print(f"图表已保存至：\n{pdf_path}\n{png_path}")
    plt.show()

if __name__ == '__main__':
    # DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    # print(f"当前目录: {DIR_PATH}")
    RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"baseline_results")
    plot_runtime_comparison("baseline_3-regular-graph.csv")
    plot_runtime_comparison("baseline_4-regular-graph.csv")
    plot_runtime_comparison("baseline_5-regular-graph.csv")
    plot_memory_comparison("baseline_3-regular-graph.csv")
    plot_memory_comparison("baseline_4-regular-graph.csv")
    plot_memory_comparison("baseline_5-regular-graph.csv")

    epsilon = 0.05  # 误差范围
    CHECK_RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"check_results")

    print_check_result(CHECK_RESULT_PATH, "baseline_2-regular-graph.csv")
    print_check_result(CHECK_RESULT_PATH, "baseline_3-regular-graph.csv")
    print_check_result(CHECK_RESULT_PATH, "baseline_4-regular-graph.csv")






