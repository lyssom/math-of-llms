import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# ----------------------------
# 画风设置
# ----------------------------
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False
bg_color = "#FAF9F6"
line_colors = ["#556B2F", "#8FBC8F", "#D4A373", "#4682B4"]  # 暗橄榄 / 浅绿 / 沙色 / 钢蓝

x = np.linspace(-5,5,500)

# ----------------------------
# 定义激活函数及导数
# ----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def dsigmoid(z):
    s = sigmoid(z)
    return s*(1-s)

def tanh(z):
    return np.tanh(z)
def dtanh(z):
    return 1 - np.tanh(z)**2

def relu(z):
    return np.maximum(0, z)
def drelu(z):
    return (z > 0).astype(float)

def gelu(z):
    return 0.5 * z * (1 + erf(z/np.sqrt(2)))
def dgelu(z):
    return 0.5*(1 + erf(z/np.sqrt(2))) + (1/np.sqrt(2*np.pi)) * z * np.exp(-z**2/2)

activations = [sigmoid, tanh, relu, gelu]
derivatives = [dsigmoid, dtanh, drelu, dgelu]
labels = ["Sigmoid", "Tanh", "ReLU", "GELU"]

# ----------------------------
# 创建子图
# ----------------------------
fig, axs = plt.subplots(2, 1, figsize=(8,6), facecolor=bg_color, sharex=True)
fig.subplots_adjust(hspace=0.4)

# 上行：函数曲线
for func, color, label in zip(activations, line_colors, labels):
    axs[0].plot(x, func(x), color=color, lw=2, label=label)
axs[0].set_title("激活函数曲线", fontsize=14, color="#3D3D3D")
axs[0].grid(True, ls="--", lw=0.5, color="#DCDCDC")
axs[0].legend(frameon=False)

# 下行：导数曲线
for dfunc, color, label in zip(derivatives, line_colors, labels):
    axs[1].plot(x, dfunc(x), color=color, lw=2, label=label)
axs[1].set_title("激活函数导数", fontsize=14, color="#3D3D3D")
axs[1].grid(True, ls="--", lw=0.5, color="#DCDCDC")
axs[1].legend(frameon=False)

# 坐标美化
axs[1].set_xlabel("输入 z", fontsize=12, color="#3D3D3D")
for ax in axs:
    ax.set_facecolor(bg_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color("#888888")
    ax.spines['left'].set_color("#888888")

# 保存与显示
plt.tight_layout()
plt.savefig("fig_activation_functions.svg", facecolor=bg_color)
plt.show()
