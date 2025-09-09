import pandas as pd
from pyecharts.charts import Surface3D
from pyecharts import options as opts
from collections import defaultdict
from itertools import product

def plot_rk_3d(path):


    # 读取数据
    df = pd.read_csv(path)

    df = df[df["r"] == 0]  # 固定 r 或者用 groupby 做平均

    # 确保列类型统一
    df["domain_size"] = df["domain_size"].astype(int)
    df["k"] = df["k"].astype(int)

    # 获取去重后的 n 和 k 列表
    n_vals = sorted(df["domain_size"].dropna().astype(int).unique().tolist())
    k_vals = sorted(df["k"].dropna().astype(int).unique().tolist())

    # 建立索引映射（n、k 转换为坐标轴索引）
    n_index = {n: i for i, n in enumerate(n_vals)}
    k_index = {k: i for i, k in enumerate(k_vals)}

    # 构建 surface 数据 [i_n, i_k, time]
    data = []
    for _, row in df.iterrows():
        i = n_index[row["domain_size"]]
        j = k_index[row["k"]]
        z = row["time_sec"]
        data.append([i, j, z])

    # 绘图
    chart = (
        Surface3D()
        .add(
            series_name="Time",
            data=data,
            xaxis3d_opts=opts.Axis3DOpts(
                type_="category", name="n", data=[str(n) for n in n_vals]
            ),
            yaxis3d_opts=opts.Axis3DOpts(
                type_="category", name="k", data=[str(k) for k in k_vals]
            ),
            zaxis3d_opts=opts.Axis3DOpts(name="Time (s)", type_="value"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Time vs n and k"),
            visualmap_opts=opts.VisualMapOpts(
                min_=df["time_sec"].min(),
                max_=df["time_sec"].max(),
                dimension=2,
            ),
        )
    )

    chart.render(path.replace('.csv', '_n_k.html'))
    print("✅ 已生成交互式 Surface 图：time_surface_n_k.html")


if __name__ == '__main__':
    path = "/home/sunshixin/pycharm_workspace/baseline/modk/results_20250807_124650/binary/binary.csv"
    plot_rk_3d(path)