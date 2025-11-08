import os

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker


def plot_correctness(csv_file):
    global RESULTS_PATH  # 获取实验结果路径
    csv_path = os.path.join(RESULTS_PATH, csv_file)  # 构造CSV文件的完整路径
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
    ouralgo_value = pd.to_numeric(ouralgo_value, errors='coerce')  # 参数 errors='coerce'：当遇到无法转换为数值的数据时（如字符串 "TIMEOUT"），将其转换为 NaN（Not a Number），而不是抛出异常
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
    mask_ouralgo = (ouralgo_value >= 1) & (ouralgo_domain <= 10)
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
    plt.figure(figsize=(12, 8), dpi=300)

    if not ouralgo_value.empty:
        plt.plot(ouralgo_domain, ouralgo_value, label="Ours", color='#ff7f0e', linewidth=3, zorder = 1)
    if not ganak_values.empty:
        plt.scatter(ganak_domain, ganak_values, color='#1f77b4', marker='s', s=200, linewidths=1, edgecolor = "black", label="Ganak (Exact)", zorder = 2)
    if not approx_values.empty:
        plt.errorbar(approx_domain, approx_values,
                     yerr=[approx_values - approx_lower, approx_upper - approx_values],
                     fmt='o', color='#2ca02c', ecolor='red', elinewidth=3, capsize=5, markersize = 8, markeredgewidth=1,  label="ApproxMC (95% CI)", zorder = 3)

    # 5. 坐标轴 & 样式
    plt.yscale("log")
    plt.xlabel("Domain Size", fontsize=20)
    plt.ylabel("Model Count (log scale)", fontsize=20)
    # plt.title("Model Counting Comparison", fontsize=16)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
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

def plot_performance(
    csv_file_path: str, metric: str = 'time_sec', ylabel: str = "Runtime (s)", output_suffix: str = "time_comparison", repeated_run_time: int = 1
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

    # 过滤掉状态为 "timeout" 的数据
    if (not os.path.exists(csv_file_path)) or os.path.getsize(
        csv_file_path
    ) == 0:  # 检查CSV文件是否存在或为空，如果是则跳过绘图
        print(f"[skip] CSV 为空，跳过绘图: {csv_file_path}")
        return
    try:
        data = pd.read_csv(csv_file_path)  # 使用pandas读取CSV文件
    except pd.errors.EmptyDataError:  # 如果CSV文件只有头部没有内容，pandas会报错
        print(f"[skip] CSV 无内容可解析，跳过绘图: {csv_file_path}")
        return
    data = data[
        ~data["status"].isin(["timeout", "error"])
    ]  # 过滤掉状态为 "timeout" 或 "error" 的数据行，这些数据不参与绘图

    # 提取数据
    algorithms = data["algorithm"].unique()  # 获取所有不重复的算法名称
    domain_sizes = sorted(data["domain_size"].unique())  # 获取所有不重复的域大小并排序
    # 创建一个新的图表，设置尺寸
    plt.figure(figsize=(12, 8))

    # 定义图例中各个算法的显示名称
    legend_labels = {"dr": "Ours", "ganak": "Ganak", "approxmc": "Approxmc"}

    # 定义不同算法在图上的标记样式
    markers = {
        "dr": "o",  # 圆形标记
        "ganak": "s",  # 正方形标记
        "approxmc": "^",  # 三角形标记
    }
    # 遍历每一种算法，分别绘制它们的性能曲线
    for algo in algorithms:
        algo_data = data[data["algorithm"] == algo]  # 筛选出当前算法的数据
        
        # 根据算法确定 domain_sizes
        if algo == "dr":
            # 为 "incremental" 算法指定 domain size
            domain_sizes = list(range(2, 20, 1))
        else:
            # 其他算法使用其在数据中存在的 domain size
            domain_sizes = sorted(algo_data['domain_size'].unique())
            
            
        values = [
            algo_data[algo_data["domain_size"] == size][metric].mean()
            for size in domain_sizes]


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

    if metric == "time_sec":  # 如果绘制的是时间，Y轴使用对数刻度
        plt.yscale("log")

    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # 设置X轴刻度为整数

    plt.xlabel("Domain Size", fontsize=18)  # 设置X轴标签和字体大小
    plt.ylabel(ylabel, fontsize=18)  # 设置Y轴标签和字体大小
    # plt.legend(fontsize=18, loc='upper left')
    plt.legend(fontsize=18)  # 显示图例
    plt.xticks(fontsize=18)  # 设置X轴刻度字体大小
    plt.yticks(fontsize=18)  # 设置Y轴刻度字体大小
    plt.grid(True, linestyle="--", alpha=0.7)  # 显示网格线

    # 保存图片到本地
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
 


if __name__ == '__main__':
    epsilon = 0.05  # 假设误差范围为 ±5%
    # RESULTS_PATH = "/home/sunshixin/pycharm_workspace/experiment/baseline/check/results/regular-graphs"  # 假设实验结果存储在 results 文件夹中
    # plot_correctness("check_2-regular-graph.csv")  # 替换为实际的 CSV 文件名
    # plot_correctness("check_3-regular-graph.csv")  # 替换为实际的 CSV 文件名
    # plot_correctness("check_4-regular-graph.csv")  # 替换为实际的 CSV 文件名
    
    # plot_performance("/home/sunshixin/pycharm_workspace/experiment/baseline/check/results/rmodk-regular-graphs/baseline_0mod2-regular-graph.csv")
    # plot_performance("/home/sunshixin/pycharm_workspace/experiment/baseline/check/results/rmodk-regular-graphs/baseline_1mod2-regular-graph.csv")
    plot_performance("/home/sunshixin/pycharm_workspace/experiment/baseline/check/results/rmodk-regular-graphs/baseline_2mod4-regular-graph.csv")
