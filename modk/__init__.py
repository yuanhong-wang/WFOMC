"""

baseline/
│
├── config.py                # 存放全局路径、默认参数等配置
├── monitor.py               # 内存/时间监控工具
├── runner.py                # 封装单次实验执行逻辑
├── csv_writer.py            # 封装CSV写入
├── plot_tools.py            # 绘图工具（从原框架拆过来）
└── run_k_r_experiment.py    # 具体的 n,k,r 遍历实验入口
"""