import os

import pandas as pd
from matplotlib import pyplot as plt


def print_result(csv_file):
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

if __name__ == '__main__':
    epsilon = 0.05  # 假设误差范围为 ±5%
    RESULTS_PATH = "/home/sunshixin/pycharm_workspace/experiment/baseline/check/results/regular-graphs"  # 假设实验结果存储在 results 文件夹中
    print_result("check_2-regular-graph.csv")  # 替换为实际的 CSV 文件名
    print_result("check_3-regular-graph.csv")  # 替换为实际的 CSV 文件名
    print_result("check_4-regular-graph.csv")  # 替换为实际的 CSV 文件名
