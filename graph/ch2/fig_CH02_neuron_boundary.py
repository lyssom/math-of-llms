import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 设置画风
# ----------------------------
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False
bg_color = "#FAF9F6"
line_color = "#556B2F"  # 暗橄榄
active_color = "#8FBC8F"
inactive_color = "#D4A373"

plt.figure(figsize=(6,6), facecolor=bg_color)

# ----------------------------
# 定义神经元参数
# ----------------------------
w = np.array([1.0, -1.0])
b = 0.2

def sigma(z):
    return 1 / (1 + np.exp(-z))

# ----------------------------
# 网格
# ----------------------------
x1 = np.linspace(-3,3,200)
x2 = np.linspace(-3,3,200)
X1, X2 = np.meshgrid(x1, x2)
Z = w[0]*X1 + w[1]*X2 + b
Y = sigma(Z)

# ----------------------------
# 绘制激活前等值线 (线性组合 z)
# ----------------------------
contour_z = plt.contour(X1, X2, Z, levels=[0], colors=line_color, linewidths=2)
plt.clabel(contour_z, fmt="决策边界 z=0", inline=True, fontsize=10, colors=line_color)

# ----------------------------
# 绘制激活后热力图 y = σ(z)
# ----------------------------
plt.contourf(X1, X2, Y, levels=50, cmap=plt.get_cmap('Greens'), alpha=0.1)

# ----------------------------
# 输入点示例
# ----------------------------
points = np.array([[1,2], [-1,1], [2,-2], [-2,-1]])
for x_pt in points:
    z_pt = np.dot(w, x_pt) + b
    y_pt = sigma(z_pt)
    color = active_color if y_pt>0.5 else inactive_color
    plt.scatter(x_pt[0], x_pt[1], s=80, color=color, edgecolor=line_color, linewidth=1.5)
    plt.text(x_pt[0]+0.1, x_pt[1]+0.1, f"y={y_pt:.2f}", fontsize=10, color="#3D3D3D")

# ----------------------------
# 坐标轴与标题
# ----------------------------
plt.title("单个神经元几何解释", fontsize=14, color="#3D3D3D")
plt.grid(True, ls="--", lw=0.5, color="#DCDCDC")

# ----------------------------
# 隐藏坐标轴
# ----------------------------
plt.xticks([])  # 去掉 x 轴刻度
plt.yticks([])  # 去掉 y 轴刻度
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.xlim([-3,3])
plt.ylim([-3,3])
plt.gca().set_aspect('equal', adjustable='box')

# ----------------------------
# 保存
# ----------------------------
plt.tight_layout()
plt.savefig("fig_单神经元_几何解释_无坐标.svg", facecolor=bg_color)
plt.show()
