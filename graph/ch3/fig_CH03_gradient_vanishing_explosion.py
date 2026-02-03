import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 画风设置
# ----------------------------
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False
bg_color = "#FAF9F6"
line_colors = ["#556B2F", "#8FBC8F", "#D4A373"]  # 不同情景

plt.figure(figsize=(7,5), facecolor=bg_color)

# ----------------------------
# 网络层数
# ----------------------------
L = 50  # 层数
layers = np.arange(1, L+1)

# ----------------------------
# 不同 ∥Wk∥ 情景
# ----------------------------
W_small = 0.9  # 梯度消失
W_unit = 1.0   # 稳定
W_large = 1.1  # 梯度爆炸

grad_decay = W_small ** layers
grad_stable = W_unit ** layers
grad_explode = W_large ** layers

# ----------------------------
# 绘制对数坐标曲线
# ----------------------------
plt.plot(layers, grad_decay, label="梯度消失 ∏‖Wk‖<1", color=line_colors[0], lw=2)
plt.plot(layers, grad_stable, label="梯度稳定 ∏‖Wk‖≈1", color=line_colors[1], lw=2)
plt.plot(layers, grad_explode, label="梯度爆炸 ∏‖Wk‖>1", color=line_colors[2], lw=2)

plt.yscale('log')  # 对数坐标

# ----------------------------
# 标注
# ----------------------------
plt.xlabel("网络层数 L", fontsize=12, color="#3D3D3D")
plt.ylabel("梯度幅值 ∏k‖Wk‖", fontsize=12, color="#3D3D3D")
plt.title("多层网络梯度随层数变化（梯度消失 & 爆炸）", fontsize=14, color="#3D3D3D")
plt.grid(True, ls="--", lw=0.5, color="#DCDCDC")
plt.legend(frameon=False)

# ----------------------------
# 保存与显示
# ----------------------------
plt.tight_layout()
plt.savefig("fig_gradient_vanishing_exploding.svg", facecolor=bg_color)
plt.show()
