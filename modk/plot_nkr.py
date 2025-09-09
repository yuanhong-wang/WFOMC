import pandas as pd
from pyecharts.charts import Scatter3D
from pyecharts import options as opts
from pyecharts.globals import ThemeType


def build_named_3d_data(df: pd.DataFrame) -> list[dict]:
    """
    将 DataFrame 转换为 pyecharts 3D 图所需的结构化数据。

    每个数据点为：
        {
            "value": [n, k, r, time],
            "name": "n=.., k=.., r=..\ntime=..s"
        }

    :param df: 包含 domain_size, k, r, time_sec 列的 DataFrame
    :return: List of dicts 可用于 Scatter3D
    """
    data = []
    for _, row in df.iterrows():
        n = int(row["domain_size"])
        k = int(row["k"])
        r = int(row["r"])
        t = float(row["time_sec"])
        name = f"n={n}, k={k}, r={r}\ntime={t:.2f}s"
        data.append({
            "value": [n, k, r, t],
            "name": name
        })
    return data


def plot_nkr(path):
    # 读取 CSV
    df = pd.read_csv(path)

    # 可选：只看某个算法
    df = df[df["algorithm"] == "dr"]

    # 生成结构化数据
    data = build_named_3d_data(df)

    # 构建 3D 散点图
    scatter = (
        Scatter3D(init_opts=opts.InitOpts(width="900px", height="600px", theme=ThemeType.LIGHT))
        .add(
            series_name="运行时间",
            data=data,
            xaxis3d_opts=opts.Axis3DOpts(name="n", type_="value"),
            yaxis3d_opts=opts.Axis3DOpts(name="k", type_="value"),
            zaxis3d_opts=opts.Axis3DOpts(name="r", type_="value"),
            grid3d_opts=opts.Grid3DOpts(width=100, depth=100, height=80),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Time vs n, k, r (3D)"),
            visualmap_opts=opts.VisualMapOpts(
                dimension=3,  # 用 time_sec 映射颜色
                min_=df["time_sec"].min(),
                max_=df["time_sec"].max(),
                range_color=["#50a3ba", "#eac763", "#d94e5d"],
            ),
            tooltip_opts=opts.TooltipOpts(formatter="{name}"),
        )
    )

    # 渲染为 HTML
    scatter.render(path.replace('.csv', '_nkr.html'))
    print("✅ 已生成交互图：interactive_time_3d.html（双击可旋转查看）")

if __name__ == '__main__':
    path = "/home/sunshixin/pycharm_workspace/baseline/modk/results_20250807_124650/binary/binary.csv"
    plot_nkr(path)