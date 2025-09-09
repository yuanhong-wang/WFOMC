import matplotlib.pyplot as plt
import networkx as nx

# 创建旧模型 μ' 图
G_old = nx.Graph()
old_nodes = [1, 2, 3]
G_old.add_nodes_from(old_nodes)
G_old.add_edges_from([(1, 2), (2, 3), (1, 3)])  # 原有边

# 创建新模型 μ 图
G_new = nx.Graph()
new_nodes = [1, 2, 3, 4]  # 添加新元素 h=4
G_new.add_nodes_from(new_nodes)
G_new.add_edges_from([(1, 2), (2, 3), (1, 3)])  # 原有边
# 新节点的边（2-tables）
G_new.add_edges_from([(4, 1), (4, 2), (4, 3)])

# 布局（保证两图一致）
pos = nx.spring_layout(G_new, seed=42)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 旧模型 μ'
nx.draw_networkx_nodes(G_old, pos, node_color=['lightblue', 'lightgreen', 'lightpink'], ax=axes[0])
nx.draw_networkx_labels(G_old, pos, labels={n: str(n) for n in old_nodes}, ax=axes[0])
nx.draw_networkx_edges(G_old, pos, ax=axes[0])
axes[0].set_title("Old model μ' (domain [h-1])")
axes[0].axis('off')

# 新模型 μ
node_colors_new = ['lightblue', 'lightgreen', 'lightpink', 'orange']  # 新节点用橙色表示新1-type
nx.draw_networkx_nodes(G_new, pos, node_color=node_colors_new, ax=axes[1])
nx.draw_networkx_labels(G_new, pos, labels={n: str(n) for n in new_nodes}, ax=axes[1])
# 原有边为黑色，新边为红色虚线
nx.draw_networkx_edges(G_new, pos, edgelist=[(1, 2), (2, 3), (1, 3)], edge_color='black', ax=axes[1])
nx.draw_networkx_edges(G_new, pos, edgelist=[(4, 1), (4, 2), (4, 3)], edge_color='red', style='dashed', ax=axes[1])
axes[1].set_title("New model μ (domain [h])")
axes[1].axis('off')

plt.suptitle("Extending μ' to μ by adding new element h", fontsize=14)
plt.tight_layout()
plt.savefig("extend_model.png", dpi=300)
plt.show()
