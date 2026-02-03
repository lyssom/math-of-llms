import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
model_size = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9])
loss = 2.0 * model_size**(-0.07) + 0.05

# 颜色和风格
line_color = "#556B2F"       # 暗橄榄绿，柔和自然
marker_color = "#8FBC8F"     # 浅绿色
bg_color = "#FAF9F6"         # 米白背景
grid_color = "#DCDCDC"       # 淡灰色网格
font_family = "serif"

# 创建图
plt.figure(figsize=(8,6), facecolor=bg_color)
plt.loglog(model_size, loss, marker='o', linewidth=2, markersize=8, color=line_color, markerfacecolor=marker_color, markeredgewidth=0.5)

# 留白：只标注关键点
for x, y in zip(model_size[[0,3,-1]], loss[[0,3,-1]]):
    plt.text(x*1.1, y*1.05, f'{int(x/1e6)}M', fontsize=10, fontfamily=font_family, color="#555555")

# 坐标轴和标题
plt.xlabel("Model Size (Parameters)", fontsize=12, fontfamily=font_family, color="#333333")
plt.ylabel("Loss", fontsize=12, fontfamily=font_family, color="#333333")
plt.title("Scaling Law: Loss vs Model Size", fontsize=14, fontfamily=font_family, color="#333333")

# 网格与边框
plt.grid(True, which="both", ls="--", lw=0.5, color=grid_color)
plt.tick_params(colors="#555555")  # 坐标轴刻度颜色
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 保存
plt.tight_layout()
plt.savefig("fig_CH10_scaling_law_wabisabi.svg", facecolor=bg_color)
plt.show()
