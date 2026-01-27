import numpy as np
import matplotlib.pyplot as plt

# ---------- 中文字体设置 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 输入范围
x = np.linspace(-6, 6, 1000)

# 激活函数及其导数
sigmoid = 1 / (1 + np.exp(-x))
sigmoid_deriv = sigmoid * (1 - sigmoid)

tanh = np.tanh(x)
tanh_deriv = 1 - tanh**2

relu = np.maximum(0, x)
relu_deriv = (x > 0).astype(float)  # ReLU导数：0或1

# 绘图
plt.figure(figsize=(8,5))

plt.plot(x, sigmoid_deriv, label='Sigmoid 导数', color='blue')
plt.plot(x, tanh_deriv, label='Tanh 导数', color='green')
plt.plot(x, relu_deriv, label='ReLU 导数', color='red')

plt.xlabel('输入 x')
plt.ylabel('导数值')
plt.title('图9.2.2 不同激活函数导数分布对比图')
plt.grid(True, ls='--', lw=0.5)
plt.legend()
plt.show()
