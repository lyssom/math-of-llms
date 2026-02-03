import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 画风
# ----------------------------
plt.rcParams['font.family'] = ['Microsoft YaHei']
bg_color = "#FAF9F6"
line_color = "#556B2F"

plt.figure(figsize=(6,4), facecolor=bg_color)

# ----------------------------
# 模拟注意力矩阵奇异值
# ----------------------------
num_singular = 50
singular_values = np.exp(-0.1*np.arange(num_singular)) + 0.02*np.random.rand(num_singular)

# ----------------------------
# STEM 绘制（去掉 use_line_collection）
# ----------------------------
markerline, stemlines, baseline = plt.stem(
    np.arange(1, num_singular+1), singular_values,
    linefmt=line_color, markerfmt="o", basefmt=" "
)

# ----------------------------
# 高级美化
# ----------------------------
plt.yscale('log')
plt.xlabel("奇异值序号", fontsize=12)
plt.ylabel("奇异值大小 (对数)", fontsize=12)
plt.title("Attention 矩阵奇异值衰减", fontsize=14, color=line_color)
plt.grid(True, which="both", ls="--", lw=0.5, color="#DCDCDC")
plt.tight_layout()

# ----------------------------
# 保存
# ----------------------------
plt.savefig("fig_attention_singular_values_stem.svg", facecolor=bg_color)
plt.show()
