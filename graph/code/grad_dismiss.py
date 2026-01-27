import numpy as np
import matplotlib.pyplot as plt

# ---------- 中文字体设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 设置层数
L = 100
layers = np.arange(1, L+1)

# 三种权重情形
w_values = [0.9, 1.0, 1.1]  # |w| < 1, |w| = 1, |w| > 1
labels = ['|w| < 1 (消失)', '|w| = 1 (稳定)', '|w| > 1 (爆炸)']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8,5))

for w, label, color in zip(w_values, labels, colors):
    grad_norm = np.abs(w)**(layers-1)
    plt.plot(layers, grad_norm, label=label, color=color)

plt.yscale('log')  # 对数坐标
plt.xlabel('层数 L')
plt.ylabel('梯度模长 |∂L/∂w1| (对数坐标)')
plt.title('图9.2.1 梯度模长随层数变化示意图')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.show()
