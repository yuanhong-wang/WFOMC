import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 假设有数据记录
data = {
    'n': [4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7],
    'k': [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4],
    'r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    'time': [0.2, 0.25, 0.3, 0.25, 0.3, 0.35, 0.3, 0.35, 0.4, 0.35, 0.4, 0.45],
}

# 转换为 DataFrame
df = pd.DataFrame(data)

# 创建分面图
g = sns.FacetGrid(df, col="r", height=5, aspect=1.5)
g.map(sns.scatterplot, "n", "k", "time", palette="viridis", hue="time", legend=False)

# 设置标签
g.set_axis_labels("Domain Size (n)", "Modulus (k)")
g.set_titles("Remainder (r) = {col_name}")
plt.rcParams['font.sans-serif'] = ['Noto Sans SC']  # 黑体，支持中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

plt.savefig("分面图.png", dpi=300)
plt.show()