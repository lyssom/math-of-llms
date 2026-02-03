import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 矩阵尺寸
# ----------------------------
seq, d = 6, 4  # 示例序列长度和隐藏维度

# 随机矩阵示意
Q = np.random.rand(seq, d)
K = np.random.rand(seq, d)
V = np.random.rand(seq, d)
QK_T = Q @ K.T
softmax = np.exp(QK_T) / np.sum(np.exp(QK_T), axis=1, keepdims=True)
attention_out = softmax @ V

# ----------------------------
# 绘图风格
# ----------------------------
bg_color = "#FAF9F6"
fig, axes = plt.subplots(1, 5, figsize=(15,3), facecolor=bg_color)

matrices = [Q, K, V, softmax, attention_out]
titles = ["Q (seq×d)", "K (seq×d)", "V (seq×d)", "Softmax(QKᵀ) (seq×seq)", "Attention Output (seq×d)"]
cmaps = ["Blues", "Blues", "Blues", "Oranges", "Greens"]

for ax, mat, title, cmap in zip(axes, matrices, titles, cmaps):
    im = ax.imshow(mat, cmap=cmap, aspect='auto')
    ax.set_title(title, fontsize=12, fontfamily='serif')
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', fontsize=8, color='black')

plt.tight_layout()
plt.savefig("fig_CH05_attention_matrix_blocks_wabisabi.svg", facecolor=bg_color)
plt.show()
