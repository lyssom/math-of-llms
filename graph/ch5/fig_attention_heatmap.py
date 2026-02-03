import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 画风设置
# ----------------------------
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False
bg_color = "#FAF9F6"
plt.figure(figsize=(6,6), facecolor=bg_color)

# ----------------------------
# 随机 attention 分数矩阵
# ----------------------------
np.random.seed(42)
N = 8  # token 数量
scores = np.random.randn(N, N)

# 行归一化 (softmax)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

attention = softmax(scores)

# ----------------------------
# 绘制热力图
# ----------------------------
im = plt.imshow(attention, cmap='Greens', vmin=0, vmax=1)

# ----------------------------
# 数值标注
# ----------------------------
for i in range(N):
    for j in range(N):
        plt.text(j, i, f"{attention[i,j]:.2f}",
                 ha='center', va='center', color="#3D3D3D", fontsize=10)

# ----------------------------
# 坐标轴
# ----------------------------
plt.xticks(range(N), [f"Token {i+1}" for i in range(N)], rotation=45, fontsize=10)
plt.yticks(range(N), [f"Token {i+1}" for i in range(N)], fontsize=10)
plt.title("注意力权重概率热力图 (行归一化)", fontsize=14, color="#3D3D3D")
plt.grid(False)

# ----------------------------
# 色条
# ----------------------------
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("Attention 权重", fontsize=12)

# ----------------------------
# 保存
# ----------------------------
plt.tight_layout()
plt.savefig("fig_attention_heatmap.svg", facecolor=bg_color)
plt.show()
