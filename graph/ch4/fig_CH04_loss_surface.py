import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 输入空间模拟
# ----------------------------
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)

# ----------------------------
# MSE Loss: L = 0.5 * ((x-0.5)^2 + (y-0.5)^2)
# ----------------------------
Z_mse = 0.5 * ((X-0.5)**2 + (Y-0.5)**2)

# ----------------------------
# Cross-Entropy Loss (二分类 softmax)
# L = -y_true*log(sigmoid(x)) - (1-y_true)*log(1-sigmoid(x))
# 模拟 target = 1, 2D 输入
# ----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Z_ce = -np.log(sigmoid(X)) - np.log(sigmoid(Y))  # 简化 2D

# ----------------------------
# 画图设置
# ----------------------------
bg_color = "#FAF9F6"      # 米白背景
grid_color = "#DCDCDC"
font_family = "serif"

fig, axes = plt.subplots(1, 2, figsize=(12,5), facecolor=bg_color)

# -------- MSE 曲面 + 等高线 --------
ax = axes[0]
cs = ax.contourf(X, Y, Z_mse, levels=20, cmap='Blues', alpha=0.6)
ax.contour(X, Y, Z_mse, levels=10, colors='#556B2F', linewidths=0.8)
ax.set_title("MSE Loss Landscape", fontsize=14, fontfamily=font_family)
ax.set_xlabel("x", fontsize=12, fontfamily=font_family)
ax.set_ylabel("y", fontsize=12, fontfamily=font_family)

# -------- CE 曲面 + 等高线 --------
ax = axes[1]
cs = ax.contourf(X, Y, Z_ce, levels=20, cmap='Reds', alpha=0.6)
ax.contour(X, Y, Z_ce, levels=10, colors='#A0522D', linewidths=0.8)
ax.set_title("Cross-Entropy Loss Landscape", fontsize=14, fontfamily=font_family)
ax.set_xlabel("x", fontsize=12, fontfamily=font_family)
ax.set_ylabel("y", fontsize=12, fontfamily=font_family)

# 网格和风格
for ax in axes:
    ax.set_facecolor(bg_color)
    ax.grid(False)

# 保存
plt.tight_layout()
plt.savefig("fig_CH04_loss_surfaces_wabisabi.svg", facecolor=bg_color)
plt.show()
