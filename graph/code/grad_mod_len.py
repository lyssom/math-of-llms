import numpy as np
import matplotlib.pyplot as plt

# ---------- 中文字体设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False    # 负号显示正常

# 层数
L = 100
layers = np.arange(1, L+1)

# 不同alpha值
alpha_values = [0.9, 1.0, 1.1]  # <1 消失, =1 稳定, >1 爆炸
labels = [r'$\alpha < 1$ (梯度消失)', r'$\alpha = 1$ (稳定)', r'$\alpha > 1$ (梯度爆炸)']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8,5))

for alpha, label, color in zip(alpha_values, labels, colors):
    grad_norm = alpha**(layers-1)
    plt.plot(layers, grad_norm, label=label, color=color)

plt.yscale('log')  # 对数坐标
plt.xlabel('层数 L')
plt.ylabel(r'梯度范数 $\|\mathbf{J}^L\|$ (对数坐标)')
plt.title('图9.2.3 梯度模长随深度变化曲线（对数坐标）')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.show()
