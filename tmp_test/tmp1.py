import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(8,5))

# 大外框
ax.add_patch(patches.Rectangle((0,0), 8, 5, fill=False, linewidth=2))
ax.text(4, 5.2, r"$\mathcal{M}_{\Gamma,h}$", ha='center', fontsize=14)

# 模拟三个 μ'，每个有两个 l
mu_primes = ["$\mu'_1$", "$\mu'_2$", "$\mu'_3$"]
l_vals = ["$\\tau_1$", "$\\tau_2$"]
block_w, block_h = 2.5, 2.2

for i, mu in enumerate(mu_primes):
    for j, l in enumerate(l_vals):
        x = 0.3 + i*block_w
        y = 0.3 + j*block_h
        ax.add_patch(patches.Rectangle((x,y), block_w-0.6, block_h-0.6,
                                       fill=True, alpha=0.2, edgecolor='black'))
        ax.text(x+ (block_w-0.6)/2, y+(block_h-0.6)/2,
                f"{mu} + {l}", ha='center', va='center', fontsize=12)

# 去掉坐标轴
ax.set_xlim(0,8)
ax.set_ylim(0,5)
ax.axis('off')

plt.tight_layout()
plt.savefig("partition_M_mu_l.png", dpi=300)
plt.show()
