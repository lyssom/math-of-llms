"""
linear_algebra_wabisabi.py
高端诧寂风单图：线性代数几何直觉（向量、矩阵与 Ax=b）
特点：
- 米白背景，灰黑渐变箭头
- 暗橄榄绿突出 Ax 投影
- 中文注释，ASCII 下标避免字体缺失
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 高端诧寂风统一设置
# ----------------------------
bg_color = "#FAF9F6"           # 米白背景
font_family = "SimSun"         # 中文字体
plt.rcParams["font.family"] = font_family
plt.rcParams["axes.unicode_minus"] = False  # 保证负号显示正常

# ----------------------------
# 基本矩阵和向量
# ----------------------------
A = np.array([[2, 1],
              [0.5, 1]])
v = np.array([1.5, 0.8])
b = np.array([1.8, 1.0])
x_candidate = np.linalg.pinv(A) @ b
b_est = A @ x_candidate

# 单位圆
theta = np.linspace(0, 2*np.pi, 400)
circle = np.vstack((np.cos(theta), np.sin(theta)))
ellipse = A @ circle

# 矩阵列 = 基向量变换
e1_t = A @ np.array([1,0])
e2_t = A @ np.array([0,1])

# ----------------------------
# 绘图
# ----------------------------
fig, ax = plt.subplots(figsize=(6,6), facecolor=bg_color)
ax.set_aspect('equal')
ax.axis('off')  # 留白感

# 单位圆（淡灰虚线）
ax.plot(circle[0], circle[1], linestyle='--', color='#B0B0B0', alpha=0.5, linewidth=1)

# 椭圆（矩阵变换结果）
ax.plot(ellipse[0], ellipse[1], color="#333333", alpha=0.9, linewidth=2)

# 渐变射线增加空间感
for i in range(0, len(ellipse[0]), 15):
    ax.plot([0, ellipse[0][i]], [0, ellipse[1][i]], color="#333333", alpha=0.05)

# 原基向量（淡灰）
ax.arrow(0,0,1,0, head_width=0.05, color='dimgray', alpha=0.4)
ax.arrow(0,0,0,1, head_width=0.05, color='dimgray', alpha=0.4)

# 变换后的基向量（黑色）
ax.arrow(0,0,e1_t[0],e1_t[1], head_width=0.05, color='black', alpha=0.8)
ax.arrow(0,0,e2_t[0],e2_t[1], head_width=0.05, color='black', alpha=0.8)
ax.text(e1_t[0]+0.15, e1_t[1], 'A e1', fontsize=10, color="#333333")
ax.text(e2_t[0]+0.15, e2_t[1], 'A e2', fontsize=10, color="#333333")

# 单个向量 v
ax.arrow(0,0,v[0],v[1], head_width=0.05, color='black', alpha=0.9)
ax.text(v[0]+0.05, v[1], '向量 v', fontsize=10, color="#333333")

# b 和 Ax 投影
ax.arrow(0,0,b[0],b[1], head_width=0.05, color='black', alpha=0.9)
ax.text(b[0]+0.05, b[1], '向量 b', fontsize=10, color="#333333")

# Ax 用暗橄榄绿强调
ax.arrow(0,0,b_est[0],b_est[1], head_width=0.05, color='#556B2F', alpha=0.8, linestyle='dashed')
# ax.text(b_est[0]+0.05, b_est[1], 'Ax', fontsize=10, color="#556B2F")

# 标题
ax.set_title("线性代数几何直觉（向量、矩阵与 Ax=b）", 
             fontsize=14, color="#333333", pad=20)

# 保存 SVG
plt.tight_layout()
plt.savefig("fig_线性代数_单图_暗橄榄强调.svg", facecolor=bg_color)
plt.show()
