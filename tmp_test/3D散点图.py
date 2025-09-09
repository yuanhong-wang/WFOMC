import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设有实验结果数据
n_values = [4, 5, 6, 7]
k_values = [2, 3, 4]
r_values = [0, 1]

times = [
    0.2, 0.3, 0.4, 0.5,  # Corresponding to n=4
    0.25, 0.35, 0.45, 0.55,  # Corresponding to n=5
    0.3, 0.4, 0.5, 0.6,  # Corresponding to n=6
    0.35, 0.45, 0.55, 0.65  # Corresponding to n=7
]

# Ensure you have enough values in 'times'
expected_size = len(n_values) * len(k_values) * len(r_values)

if len(times) < expected_size:
    print(f"Warning: Not enough time values. Expected {expected_size}, but got {len(times)}.")
    # You could either truncate or repeat the times list to match the required size
    times = times * (expected_size // len(times)) + times[:expected_size % len(times)]

# Now, you can safely proceed to append the times without IndexError
n_expanded = []
k_expanded = []
r_expanded = []
time_expanded = []

for n in n_values:
    for k in k_values:
        for r in r_values:
            n_expanded.append(n)
            k_expanded.append(k)
            r_expanded.append(r)
            time_expanded.append(times.pop(0))  # Now times should have enough values


# 创建 3D 散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点
sc = ax.scatter(n_expanded, k_expanded, r_expanded, c=time_expanded, cmap='viridis', s=100)

# 添加标签
ax.set_xlabel('Domain Size (n)')
ax.set_ylabel('Modulus (k)')
ax.set_zlabel('Remainder (r)')
ax.set_title('3D Scatter Plot of Time vs n, k, and r')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，支持中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['font.sans-serif'] = ['Noto Sans SC']  # 黑体，支持中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.savefig("3D散点图.png", dpi=300)
# 添加颜色条
plt.colorbar(sc, label='Time (s)')

plt.show()
