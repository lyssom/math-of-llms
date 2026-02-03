import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# ----------------------------
# 模拟 Hessian / Jacobian 特征值
# ----------------------------
np.random.seed(42)

# 样本 1: 标准初始化
eigvals1 = np.random.normal(0, 0.5, 200) + 1j*np.random.normal(0, 0.2, 200)

# 样本 2: 带轻微正则化
eigvals2 = np.random.normal(0, 0.4, 200) + 1j*np.random.normal(0, 0.15, 200)

# 样本 3: 强正则化
eigvals3 = np.random.normal(0, 0.3, 200) + 1j*np.random.normal(0, 0.1, 200)

samples = [eigvals1, eigvals2, eigvals3]
labels = ["Init", "Light Reg", "Strong Reg"]
colors = ["#8FBC8F", "#556B2F", "#A0522D"]  # 柔和绿 / 暗橄榄 / 土色

# ----------------------------
# 绘图设置
# ----------------------------
bg_color = "#FAF9F6"  # 米白
grid_color = "#DCDCDC"
font_family = "serif"

plt.figure(figsize=(8,6), facecolor=bg_color)

for eigvals, label, color in zip(samples, labels, colors):
    # 取实部和虚部
    x = eigvals.real
    y = eigvals.imag

    # 核密度估计
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # 散点 + 密度色彩
    plt.scatter(x, y, c=z, s=30, cmap='Greens', alpha=0.6, edgecolor=color, label=label)

# ----------------------------
# 坐标轴与标题
# ----------------------------
plt.xlabel("Real Part", fontsize=12, fontfamily=font_family, color="#333333")
plt.ylabel("Imag Part", fontsize=12, fontfamily=font_family, color="#333333")
plt.title("Jacobian / Hessian Eigenvalue Spectrum (with Regularization)", fontsize=14, fontfamily=font_family, color="#333333")

# 网格与边框
plt.grid(True, ls="--", lw=0.5, color=grid_color)
plt.tick_params(colors="#555555")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 图例
plt.legend(frameon=False, fontsize=10, loc='upper right')

# 保存
plt.tight_layout()
plt.savefig("fig_CH09_jacobian_hessian_spectrum_wabisabi_upgrade.svg", facecolor=bg_color)
plt.show()
