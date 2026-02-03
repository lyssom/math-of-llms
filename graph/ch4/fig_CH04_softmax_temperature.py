import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Softmax 函数
# ----------------------------
def softmax(logits, tau=1.0):
    exp_logits = np.exp(logits / tau)
    return exp_logits / np.sum(exp_logits)

# 模拟 logits
logits = np.linspace(-3, 3, 100)

# 不同温度参数
taus = [0.5, 1.0, 2.0]
colors = ["#556B2F", "#8FBC8F", "#A0522D"]  # 暗橄榄、柔和绿、土色
labels = [r"$\tau=0.5$", r"$\tau=1.0$", r"$\tau=2.0$"]

# ----------------------------
# 画图设置
# ----------------------------
bg_color = "#FAF9F6"
font_family = "serif"
plt.figure(figsize=(8,6), facecolor=bg_color)

# softmax 概率 (2-class 示例)
for tau, color, label in zip(taus, colors, labels):
    prob = 1 / (1 + np.exp(-2*logits/tau))  # 2-class softmax = sigmoid
    plt.plot(logits, prob, color=color, lw=2, label=label)

# ----------------------------
# 坐标轴 & 标题
# ----------------------------
plt.xlabel("Logits", fontsize=12, fontfamily=font_family, color="#333333")
plt.ylabel("Probability", fontsize=12, fontfamily=font_family, color="#333333")
plt.title("Softmax Probabilities vs Logits (Effect of Temperature τ)", fontsize=14, fontfamily=font_family, color="#333333")

# 网格 & 风格
plt.grid(True, ls="--", lw=0.5, color="#DCDCDC")
plt.gca().set_facecolor(bg_color)
plt.tick_params(colors="#555555")
plt.legend(frameon=False, fontsize=10)

# 保存
plt.tight_layout()
plt.savefig("fig_CH04_softmax_temperature_wabisabi.svg", facecolor=bg_color)
plt.show()
