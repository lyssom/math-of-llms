import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------------
# 高端诧寂风设置
# ----------------------------
bg_color = "#FAF9F6"
font_family = "SimSun"
plt.rcParams["font.family"] = font_family
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------
# 离散分布
# ----------------------------
x_disc = np.arange(0, 6)
p_disc = np.array([0.1, 0.2, 0.3, 0.2, 0.15, 0.05])

# ----------------------------
# 连续高斯分布
# ----------------------------
x_cont = np.linspace(-5, 5, 400)
gaussians = [
    {"mu":0, "sigma":1, "color":"#8FBC8F", "label":"μ=0, σ=1"},
    {"mu":0, "sigma":1.5, "color":"#556B2F", "label":"μ=0, σ=1.5"},
    {"mu":1, "sigma":1, "color":"#A0522D", "label":"μ=1, σ=1"},
]

# ----------------------------
# 绘图
# ----------------------------
fig, ax = plt.subplots(figsize=(8,5), facecolor=bg_color)

# 离散分布柱状
ax.bar(x_disc, p_disc, width=0.4, color="#333333", alpha=0.6, label="离散分布")

# 连续分布曲线
for g in gaussians:
    y = norm.pdf(x_cont, loc=g["mu"], scale=g["sigma"])
    ax.plot(x_cont, y, color=g["color"], linewidth=2, label=f'高斯分布 {g["label"]}')

# ----------------------------
# 标注期望
# ----------------------------
E_disc = np.sum(x_disc * p_disc)
ax.axvline(E_disc, color="#333333", linestyle="--", alpha=0.7)
ax.text(E_disc+0.1, 0.05, f"期望 E[X]={E_disc:.2f}", fontsize=10, color="#333333")

# 连续分布期望
for g in gaussians:
    ax.axvline(g["mu"], color=g["color"], linestyle="--", alpha=0.5)
    ax.text(g["mu"]+0.1, 0.15, f"E[X]={g['mu']}", fontsize=10, color=g["color"])

# ----------------------------
# 风格设置
# ----------------------------
ax.set_xlabel("X", fontsize=12, color="#333333")
ax.set_ylabel("概率 / 密度", fontsize=12, color="#333333")
ax.set_title("概率分布与期望", fontsize=14, color="#333333", pad=15)
ax.grid(True, linestyle='--', linewidth=0.5, color="#DCDCDC", alpha=0.5)
ax.tick_params(colors="#555555")
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("fig_probability_expectation_wabisabi.svg", facecolor=bg_color)
plt.show()
