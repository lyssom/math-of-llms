"""
Simplified Information Geometry Visualization
按照指定配色方案：真实分布=黑，模型=蓝，优化=红，虚线=投影，点线=等高线
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon

# 设置样式
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
ax.set_facecolor('#fafafa')

# ============================================
# 概率单纯形（三角形）
# ============================================

v1 = np.array([-2.5, -1.2])
v2 = np.array([2.5, -1.2])
v3 = np.array([0, 2.0])

# 绘制单纯形
simplex = Polygon([v1, v2, v3], closed=True, 
                   facecolor='#f0f0f0', edgecolor='#888888', 
                   linewidth=2, alpha=0.5)
ax.add_patch(simplex)

# 顶点
ax.annotate('(1,0,0)', xy=v1 + np.array([-0.2, -0.3]), fontsize=10, color='#666')
ax.annotate('(0,1,0)', xy=v2 + np.array([0.1, -0.3]), fontsize=10, color='#666')
ax.annotate('(0,0,1)', xy=v3 + np.array([0, 0.2]), fontsize=10, color='#666')
ax.text(0, -0.15, 'Probability Simplex', ha='center', fontsize=9, color='#888')

# ============================================
# 等值线 / 等距线（点线）
# ============================================

q_pos = np.array([0.1, 0.4])

# 等值线（点线）
for i, (level, alpha) in enumerate([(0.15, 0.12), (0.35, 0.1), (0.6, 0.08)]):
    scale = 1 + i * 0.8
    ellipse = Ellipse(
        xy=q_pos,
        width=0.7 * scale,
        height=0.5 * scale,
        facecolor='#aaaaaa',
        edgecolor='#aaaaaa',
        linewidth=1.5,
        alpha=alpha,
        linestyle=':'
    )
    ax.add_patch(ellipse)


# KL散度标签
ax.text(-2.8, 1.3, 'KL Divergence Contours', fontsize=10, color='#aaaaaa', style='italic')

# ============================================
# 模型参数化族（蓝色实线）
# ============================================

# 一条参数化曲线（模型流形）
t_manifold = np.linspace(0, 1, 50)
manifold_x = -1.8 + t_manifold * 2.0
manifold_y = -0.5 + 0.3 * np.sin(np.pi * t_manifold) + t_manifold * 0.9
ax.plot(manifold_x, manifold_y, color='#1e90ff', linewidth=2.5, solid_capstyle='round')
ax.annotate('Model Manifold', xy=(0.4, 0.9), fontsize=10, color='#1e90ff', fontweight='bold')

# ============================================
# 真实分布（黑色实心）
# ============================================

ax.scatter(*q_pos, s=200, c='black', edgecolors='black', linewidths=2, zorder=5)
ax.annotate('Ground Truth q', xy=q_pos + np.array([0.35, 0.15]), 
            fontsize=11, color='black', fontweight='bold')

# ============================================
# 优化过程（红色虚线箭头）
# ============================================

# 从模型流形上的点到真实分布的投影
p_proj = np.array([0.1, 0.5])  # 投影点（在模型流形上）

# 优化路径（红色虚线）
ax.plot([p_proj[0], q_pos[0]], [p_proj[1], q_pos[1]], 
        color='#dc143c', linewidth=2.5, linestyle='--', dashes=(8, 4))

# 箭头（红色）
ax.annotate('', xy=q_pos - np.array([0.08, 0.06]), 
            xytext=p_proj + np.array([0.12, 0.08]),
            arrowprops=dict(arrowstyle='->', color='#dc143c', lw=2.5))

# 优化标签
ax.text(-0.3, 0.65, 'Optimization\n(Projection)', 
        fontsize=10, color='#dc143c', fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0', edgecolor='#dc143c', alpha=0.9))

# ============================================
# 投影点（在模型流形上）
# ============================================

ax.scatter(*p_proj, s=120, c='#1e90ff', edgecolors='#104e8b', linewidths=2, zorder=5)
ax.annotate('P*', xy=p_proj + np.array([0.1, 0.2]), fontsize=12, color='#1e90ff', fontweight='bold')

# ============================================
# 标题
# ============================================

ax.set_title('Information Geometry: Optimization as Projection',
             fontsize=14, fontweight='bold', pad=15)

ax.text(0, -1.75, 'Minimize KL Divergence = Project Model onto True Distribution',
        ha='center', fontsize=11, color='#333', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#666', alpha=0.9))

# ============================================
# 图例
# ============================================

legend_items = [
    ('black', 'Ground Truth', 'solid'),
    ('#1e90ff', 'Model Manifold', 'solid'),
    ('#dc143c', 'Optimization', 'dashed'),
    ('#aaaaaa', 'KL Contours', 'dotted'),
]

y_start = 2.2
for i, (color, label, style) in enumerate(legend_items):
    if style == 'solid':
        ax.plot([-3.2, -2.8], [y_start - i*0.25, y_start - i*0.25], 
                color=color, linewidth=2.5)
    elif style == 'dashed':
        ax.plot([-3.2, -2.8], [y_start - i*0.25, y_start - i*0.25], 
                color=color, linewidth=2.5, linestyle='--', dashes=(6, 3))
    elif style == 'dotted':
        ax.plot([-3.2, -2.8], [y_start - i*0.25, y_start - i*0.25], 
                color=color, linewidth=2, linestyle=':')
    ax.text(-2.6, y_start - i*0.25, label, fontsize=9, va='center', color=color)

# ============================================
# 设置
# ============================================

ax.set_xlim(-3.5, 3.2)
ax.set_ylim(-2, 2.5)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('./info_geometry_colorcoded.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
# plt.savefig('/workspace/info_geometry_colorcoded.pdf', bbox_inches='tight',
#             facecolor='white', edgecolor='none')

print("Color-coded figure saved!")
print("  - info_geometry_colorcoded.png")
print("  - info_geometry_colorcoded.pdf")
