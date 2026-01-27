import numpy as np
import matplotlib.pyplot as plt

# ---------- 中文字体 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数
theta_v = np.pi / 6      # V^T 的旋转角
theta_u = np.pi / 4      # U 的旋转角
sigma1, sigma2 = 2.5, 1.0  # 奇异值

# 构造矩阵
Vt = np.array([
    [np.cos(theta_v), -np.sin(theta_v)],
    [np.sin(theta_v),  np.cos(theta_v)]
])

Sigma = np.array([
    [sigma1, 0],
    [0, sigma2]
])

U = np.array([
    [np.cos(theta_u), -np.sin(theta_u)],
    [np.sin(theta_u),  np.cos(theta_u)]
])

# 单位圆
t = np.linspace(0, 2*np.pi, 400)
circle = np.vstack([np.cos(t), np.sin(t)])

# 逐步变换
circle_V = Vt @ circle
ellipse_S = Sigma @ circle_V
ellipse_U = U @ ellipse_S

# 绘图
plt.figure(figsize=(6,6))

plt.plot(circle[0], circle[1], '--', label='单位圆 $\\mathcal{S}^1$')
plt.plot(circle_V[0], circle_V[1], label=r'经过 $V^{\top}$ 旋转')
plt.plot(ellipse_S[0], ellipse_S[1], label=r'经过 $\Sigma$ 缩放')
plt.plot(ellipse_U[0], ellipse_U[1], label=r'经过 $U$ 旋转（最终）')

plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('SVD 的几何解释：单位圆到椭圆的变换')
plt.legend()
plt.grid(True, ls='--', lw=0.5)

plt.show()
