import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# 构造 loss landscape (非凸示例)
# ----------------------------
def loss_fn(x, y):
    return np.sin(x) * np.cos(y) + 0.1*(x**2 + y**2)

# 网格
X = np.linspace(-3, 3, 100)
Y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(X, Y)
Z = loss_fn(X, Y)

# ----------------------------
# 模拟梯度下降轨迹
# ----------------------------
trajectory = []
x, y = 2.5, 2.5  # 初始点
trajectory.append([x, y, loss_fn(x,y)])
lr = 0.1
for _ in range(30):
    # 计算梯度
    dz_dx = np.cos(x)*np.cos(y) + 0.2*x
    dz_dy = -np.sin(x)*np.sin(y) + 0.2*y
    # 梯度下降更新
    x -= lr * dz_dx
    y -= lr * dz_dy
    trajectory.append([x, y, loss_fn(x,y)])
trajectory = np.array(trajectory)

# ----------------------------
# 绘图设置
# ----------------------------
bg_color = "#FAF9F6"       # 米白背景
surface_color = "#DCE6F1"  # 柔和蓝色面
traj_color = "#556B2F"     # 暗橄榄绿轨迹
font_family = "serif"

fig = plt.figure(figsize=(9,6), facecolor=bg_color)
ax = fig.add_subplot(111, projection='3d', facecolor=bg_color)

# 绘制 loss landscape
ax.plot_surface(X, Y, Z, cmap='Blues', alpha=0.6, edgecolor='none')

# 绘制梯度下降轨迹
ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], color=traj_color, linewidth=2, marker='o')

# 起点和终点标注
ax.text(trajectory[0,0], trajectory[0,1], trajectory[0,2]+0.1, "Start", color=traj_color, fontsize=10, fontfamily=font_family)
ax.text(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2]+0.1, "End", color=traj_color, fontsize=10, fontfamily=font_family)

# 坐标轴标签
ax.set_xlabel("x", fontsize=12, fontfamily=font_family)
ax.set_ylabel("y", fontsize=12, fontfamily=font_family)
ax.set_zlabel("Loss", fontsize=12, fontfamily=font_family)
ax.set_title("Gradient Flow on Loss Landscape", fontsize=14, fontfamily=font_family)

# 美化视角
ax.view_init(elev=30, azim=45)  # 调整视角
ax.grid(False)
ax.set_facecolor(bg_color)

# 保存
plt.tight_layout()
plt.savefig("fig_CH09_gradient_flow_wabisabi.svg", facecolor=bg_color)
plt.show()
