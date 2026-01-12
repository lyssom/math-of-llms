"""
线性可分与线性不可分问题可视化
Linear Separability and XOR Problem Visualization

用于《大模型中的数学》第一章补充材料
展示:
1. 线性可分问题（Linear Separable）
2. 线性不可分问题（Linear Inseparable）
3. 异或问题（XOR Problem）
4. 多层感知机解决异或问题

作者: MiniMax Agent
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

def plot_linear_separable():
    """
    子图1: 线性可分问题
    展示两类点可以被一条直线完全分开
    """
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 设置坐标范围
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # 类别A的点（左上区域）
    class_a_x = [-1.5, -1.2, -0.8, -1.0, -1.3]
    class_a_y = [1.5, 1.2, 1.0, 0.8, 1.0]
    
    # 类别B的点（右下区域）
    class_b_x = [0.8, 1.2, 1.5, 1.0, 1.3]
    class_b_y = [-1.5, -1.2, -1.0, -0.8, -1.0]
    
    # 绘制类别A（蓝色圆点）
    ax.scatter(class_a_x, class_a_y, c='blue', s=150, 
               label='Class A', edgecolors='white', linewidth=2, zorder=5)
    
    # 绘制类别B（红色三角形）
    ax.scatter(class_b_x, class_b_y, c='red', s=150, 
               marker='^', label='Class B', edgecolors='white', linewidth=2, zorder=5)
    
    # 绘制分类超平面（虚线）
    # 直线方程: y = -x (即 x + y = 0)
    x_line = np.linspace(-2, 2, 100)
    y_line = -x_line
    ax.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')
    
    # 绘制分类区域填充
    # 区域1: x + y > 0 → Class A (蓝色半透明)
    x_fill = np.linspace(-2, 2, 100)
    y_fill_pos = np.maximum(-x_fill, -2)  # x + y > 0 的区域
    y_fill_neg = np.minimum(-x_fill, 2)   # x + y < 0 的区域
    
    # 填充Class A区域
    ax.fill_between(x_fill, y_fill_pos, 2, alpha=0.2, color='blue')
    # 填充Class B区域
    ax.fill_between(x_fill, -2, y_fill_neg, alpha=0.2, color='red')
    
    # 标题和标签
    ax.set_title('Linear Separable Problem\n(Linearly Separable)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加说明文字
    ax.text(0, 0, 'Separable\nby Line', fontsize=10, 
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_linear_inseparable():
    """
    子图2: 线性不可分问题
    展示两类点混合在一起，无法被一条直线分开
    """
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 设置坐标范围
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # 类别A的点（环形分布）
    theta = np.linspace(0, 2*np.pi, 12)
    class_a_x = 1.2 * np.cos(theta)
    class_a_y = 1.2 * np.sin(theta)
    
    # 类别B的点（中心区域）
    class_b_x = [0, 0.3, -0.3, 0.2, -0.2]
    class_b_y = [0, 0.2, 0.3, -0.2, -0.1]
    
    # 绘制类别A（蓝色圆点）
    ax.scatter(class_a_x, class_a_y, c='blue', s=150, 
               label='Class A', edgecolors='white', linewidth=2, zorder=5)
    
    # 绘制类别B（红色三角形）
    ax.scatter(class_b_x, class_b_y, c='red', s=150, 
               marker='^', label='Class B', edgecolors='white', linewidth=2, zorder=5)
    
    # 尝试绘制分类超平面（多条失败的直线）
    # 直线1: y = 0 (水平线) - 无法分开
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1.5, 0.3, 'Test 1', fontsize=8, color='gray')
    
    # 直线2: x = 0 (垂直线) - 无法分开
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(0.3, 1.5, 'Test 2', fontsize=8, color='gray')
    
    # 直线3: y = x (对角线) - 无法分开
    x_line = np.linspace(-2, 2, 100)
    ax.plot(x_line, x_line, 'gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(-1.5, -0.8, 'Test 3', fontsize=8, color='gray')
    
    # 标题和标签
    ax.set_title('Linear Inseparable Problem\n(Linearly Inseparable)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加说明文字
    ax.text(0, -1.7, 'No single line\ncan separate these!', fontsize=11, 
            ha='center', va='center', color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    return fig

def plot_xor_problem():
    """
    子图3: 异或问题（XOR Problem）
    经典的线性不可分问题
    """
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 设置坐标范围
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # XOR数据点
    # Class 0: (0,0) 和 (1,1)
    class0_x = [-1, 1]
    class0_y = [-1, 1]
    
    # Class 1: (0,1) 和 (1,0)
    class1_x = [-1, 1]
    class1_y = [1, -1]
    
    # 绘制Class 0（蓝色圆点）
    ax.scatter(class0_x, class0_y, c='blue', s=200, 
               label='Class 0', edgecolors='white', linewidth=3, zorder=5)
    
    # 绘制Class 1（红色三角形）
    ax.scatter(class1_x, class1_y, c='red', s=200, 
               marker='^', label='Class 1', edgecolors='white', linewidth=3, zorder=5)
    
    # 标注每个点
    ax.text(-1, -1.3, '(0,0)', fontsize=12, ha='center', va='top', color='blue')
    ax.text(1, -1.3, '(1,1)', fontsize=12, ha='center', va='top', color='blue')
    ax.text(-1, 1.3, '(0,1)', fontsize=12, ha='center', va='bottom', color='red')
    ax.text(1, 1.3, '(1,0)', fontsize=12, ha='center', va='bottom', color='red')
    
    # 尝试绘制分类超平面（XOR问题无法用单条直线解决）
    # 直线1: y = x - 无法分开
    x_line = np.linspace(-2, 2, 100)
    ax.plot(x_line, x_line, 'k--', linewidth=2, alpha=0.5)
    ax.text(0.5, 0.7, 'y = x\n(incorrect)', fontsize=9, color='gray')
    
    # 直线2: y = -x - 无法分开
    ax.plot(x_line, -x_line, 'k--', linewidth=2, alpha=0.5)
    ax.text(0.5, -0.7, 'y = -x\n(incorrect)', fontsize=9, color='gray')
    
    # 绘制X标记表示无法解决
    ax.text(0, 0, '✗\nCannot Separate!', fontsize=14, 
            ha='center', va='center', color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # 标题和标签
    ax.set_title('XOR Problem\n(Exclusive OR - Classic Non-linear Problem)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加问题说明
    problem_desc = """
    Truth Table:
    (0,0) → 0
    (0,1) → 1
    (1,0) → 1
    (1,1) → 0
    """
    ax.text(-1.8, -1.8, problem_desc, fontsize=9, 
            va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_xor_solution():
    """
    子图4: 多层感知机解决异或问题
    展示如何用两层神经网络解决XOR问题
    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 设置坐标范围
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # XOR数据点（与子图3相同）
    class0_x = [-1, 1]
    class0_y = [-1, 1]
    class1_x = [-1, 1]
    class1_y = [1, -1]
    
    # 绘制Class 0（蓝色圆点）
    ax.scatter(class0_x, class0_y, c='blue', s=200, 
               label='Class 0', edgecolors='white', linewidth=3, zorder=5)
    
    # 绘制Class 1（红色三角形）
    ax.scatter(class1_x, class1_y, c='red', s=200, 
               marker='^', label='Class 1', edgecolors='white', linewidth=3, zorder=5)
    
    # 绘制两个非线性决策边界（两个超平面）
    # 边界1: x₁ + x₂ = 0 (45度线)
    x_line = np.linspace(-2, 2, 100)
    ax.plot(x_line, -x_line, 'g-', linewidth=2, alpha=0.7, label='Boundary 1: x₁ + x₂ = 0')
    
    # 边界2: x₁ - x₂ = 0 (-45度线)
    ax.plot(x_line, x_line, 'm-', linewidth=2, alpha=0.7, label='Boundary 2: x₁ - x₂ = 0')
    
    # 绘制最终的决策区域（非线性边界）
    # 两个区域的交集形成XOR解
    # 区域A: x₁ + x₂ > 0 且 x₁ - x₂ > 0 → 第一象限 (1,0)
    # 区域B: x₁ + x₂ < 0 且 x₁ - x₂ < 0 → 第三象限 (0,0)
    
    # 填充正确的分类区域
    # Class 0区域 (第三象限和部分第一象限)
    theta = np.linspace(-np.pi/4, np.pi/4, 100)
    r = 1.8
    region0_x = r * np.cos(theta)
    region0_y = r * np.sin(theta)
    ax.fill_betweenx(region0_y, -2, region0_x, alpha=0.15, color='blue')
    
    # Class 1区域 (第一象限的特定区域)
    theta = np.linspace(np.pi/4, 3*np.pi/4, 100)
    region1_x = r * np.cos(theta)
    region1_y = r * np.sin(theta)
    ax.fill_betweenx(region1_y, -2, region1_x, alpha=0.15, color='red')
    
    # 标题和标签
    ax.set_title('XOR Solution with Multi-Layer Perceptron\n(Non-linear Decision Boundary)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加神经网络结构示意图
    neural_net_text = """
    MLP Architecture:
    
    Input Layer
       ↓
    Hidden Layer (2 neurons)
       ↓  
    Output Layer (1 neuron)
    
    This creates non-linear
    decision boundaries!
    """
    ax.text(-1.9, 0.5, neural_net_text, fontsize=8, 
            va='center', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 添加✓标记表示解决
    ax.text(1.5, 1.5, '✓ Solved!', fontsize=12, 
            ha='center', va='center', color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    plt.tight_layout()
    return fig

def create_combined_figure():
    """
    创建组合图：四个子图并排显示
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Linear Classification vs Non-linear Problems\n' + 
                 r'$f(x) = \mathbf{w}^T\mathbf{x} + b$ vs MLP', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 子图1: 线性可分
    ax1 = axes[0, 0]
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    
    class_a_x = [-1.5, -1.2, -0.8, -1.0, -1.3]
    class_a_y = [1.5, 1.2, 1.0, 0.8, 1.0]
    class_b_x = [0.8, 1.2, 1.5, 1.0, 1.3]
    class_b_y = [-1.5, -1.2, -1.0, -0.8, -1.0]
    
    ax1.scatter(class_a_x, class_a_y, c='blue', s=120, label='Class A', 
                edgecolors='white', linewidth=2, zorder=5)
    ax1.scatter(class_b_x, class_b_y, c='red', s=120, marker='^', 
                label='Class B', edgecolors='white', linewidth=2, zorder=5)
    
    x_line = np.linspace(-2, 2, 100)
    ax1.plot(x_line, -x_line, 'k--', linewidth=2, label='Linear Boundary')
    ax1.fill_between(x_line, -x_line, 2, alpha=0.15, color='blue')
    ax1.fill_between(x_line, -2, -x_line, alpha=0.15, color='red')
    
    ax1.set_title('(a) Linear Separable', fontsize=14, fontweight='bold')
    ax1.set_xlabel(r'$x_1$', fontsize=11)
    ax1.set_ylabel(r'$x_2$', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 线性不可分
    ax2 = axes[0, 1]
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    theta = np.linspace(0, 2*np.pi, 12)
    class_a_x = 1.3 * np.cos(theta)
    class_a_y = 1.3 * np.sin(theta)
    class_b_x = [0, 0.4, -0.4, 0.2, -0.2, 0, 0.3]
    class_b_y = [0, 0.3, 0.4, -0.3, -0.2, 0.1, -0.1]
    
    ax2.scatter(class_a_x, class_a_y, c='blue', s=120, label='Class A', 
                edgecolors='white', linewidth=2, zorder=5)
    ax2.scatter(class_b_x, class_b_y, c='red', s=120, marker='^', 
                label='Class B', edgecolors='white', linewidth=2, zorder=5)
    
    # 尝试的直线
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.plot(x_line, x_line, 'gray', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax2.set_title('(b) Linear Inseparable', fontsize=14, fontweight='bold')
    ax2.set_xlabel(r'$x_1$', fontsize=11)
    ax2.set_ylabel(r'$x_2$', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.text(0, -1.7, 'No linear boundary!', fontsize=10, ha='center', 
             color='red', fontweight='bold')
    
    # 子图3: XOR问题
    ax3 = axes[1, 0]
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    
    class0_x = [-1, 1]
    class0_y = [-1, 1]
    class1_x = [-1, 1]
    class1_y = [1, -1]
    
    ax3.scatter(class0_x, class0_y, c='blue', s=180, label='Class 0', 
                edgecolors='white', linewidth=3, zorder=5)
    ax3.scatter(class1_x, class1_y, c='red', s=180, marker='^', 
                label='Class 1', edgecolors='white', linewidth=3, zorder=5)
    
    ax3.text(-1, -1.3, '(0,0)', fontsize=11, ha='center', color='blue')
    ax3.text(1, -1.3, '(1,1)', fontsize=11, ha='center', color='blue')
    ax3.text(-1, 1.3, '(0,1)', fontsize=11, ha='center', color='red')
    ax3.text(1, 1.3, '(1,0)', fontsize=11, ha='center', color='red')
    
    ax3.plot(x_line, x_line, 'k--', linewidth=2, alpha=0.4)
    ax3.plot(x_line, -x_line, 'k--', linewidth=2, alpha=0.4)
    
    ax3.set_title('(c) XOR Problem', fontsize=14, fontweight='bold')
    ax3.set_xlabel(r'$x_1$', fontsize=11)
    ax3.set_ylabel(r'$x_2$', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    xor_text = """XOR Truth Table:
(0,0)→0  (0,1)→1
(1,0)→1  (1,1)→0"""
    ax3.text(-1.8, -1.8, xor_text, fontsize=9, va='bottom', ha='left',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # 子图4: XOR解决方案
    ax4 = axes[1, 1]
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)
    
    ax4.scatter(class0_x, class0_y, c='blue', s=180, label='Class 0', 
                edgecolors='white', linewidth=3, zorder=5)
    ax4.scatter(class1_x, class1_y, c='red', s=180, marker='^', 
                label='Class 1', edgecolors='white', linewidth=3, zorder=5)
    
    # 非线性边界
    ax4.plot(x_line, -x_line, 'g-', linewidth=2, alpha=0.7)
    ax4.plot(x_line, x_line, 'm-', linewidth=2, alpha=0.7)
    
    # 填充区域
    theta = np.linspace(-np.pi/4, np.pi/4, 100)
    r = 1.8
    region0_x = r * np.cos(theta)
    region0_y = r * np.sin(theta)
    ax4.fill_betweenx(region0_y, -2, region0_x, alpha=0.15, color='blue')
    
    ax4.set_title('(d) MLP Solution', fontsize=14, fontweight='bold')
    ax4.set_xlabel(r'$x_1$', fontsize=11)
    ax4.set_ylabel(r'$x_2$', fontsize=11)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    mlp_text = """MLP Architecture:
    
    Input → Hidden → Output
    
    Two boundaries create
    non-linear decision region!"""
    ax4.text(-1.9, 0.3, mlp_text, fontsize=8, va='center', ha='left',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig

def create_2d_hyperplane_diagram():
    """
    创建2D输入空间中的超平面分类示意图
    展示单个超平面 vs 多个超平面
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Hyperplane Classification in 2D Input Space', 
                 fontsize=16, fontweight='bold')
    
    # 子图1: 单个超平面（直线）
    ax1 = axes[0]
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    
    # 线性可分数据
    class_a_x = [-1.5, -1.0, -1.2, -0.8]
    class_a_y = [1.2, 1.5, 0.8, 1.0]
    class_b_x = [0.8, 1.2, 1.5, 1.0]
    class_b_y = [-1.2, -1.0, -0.8, -1.5]
    
    ax1.scatter(class_a_x, class_a_y, c='royalblue', s=150, 
                label='Class +1', edgecolors='white', linewidth=2, zorder=5)
    ax1.scatter(class_b_x, class_b_y, c='coral', s=150, marker='^', 
                label='Class -1', edgecolors='white', linewidth=2, zorder=5)
    
    # 超平面方程: w₁x₁ + w₂x₂ + b = 0
    # 示例: 2x₁ + x₂ = 0
    x_line = np.linspace(-2, 2, 100)
    y_line = -2 * x_line  # w₁=2, b=0
    
    ax1.plot(x_line, y_line, 'g-', linewidth=3, label='Hyperplane')
    
    # 绘制法向量
    ax1.arrow(0, 0, 0.5, 0.25, head_width=0.1, head_length=0.05, 
              fc='purple', ec='purple', linewidth=2)
    ax1.text(0.6, 0.35, r'$\mathbf{w}$', fontsize=14, color='purple')
    
    ax1.set_title('Single Hyperplane: $\mathbf{w}^T\mathbf{x} + b = 0$', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel(r'$x_1$', fontsize=12)
    ax1.set_ylabel(r'$x_2$', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax1.text(0, 0, 'Decision\nBoundary', fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 子图2: 多个超平面（MLP）
    ax2 = axes[1]
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    # XOR数据
    class0_x = [-1, 1]
    class0_y = [-1, 1]
    class1_x = [-1, 1]
    class1_y = [1, -1]
    
    ax2.scatter(class0_x, class0_y, c='royalblue', s=150, 
                label='Class 0', edgecolors='white', linewidth=2, zorder=5)
    ax2.scatter(class1_x, class1_y, c='coral', s=150, marker='^', 
                label='Class 1', edgecolors='white', linewidth=2, zorder=5)
    
    # 第一个超平面
    ax2.plot(x_line, x_line, 'b--', linewidth=2, alpha=0.7, 
             label='Hidden neuron 1')
    
    # 第二个超平面
    ax2.plot(x_line, -x_line, 'r--', linewidth=2, alpha=0.7,
             label='Hidden neuron 2')
    
    # 最终决策边界（非线性）
    # 通过两个超平面的组合形成
    region_fill = np.zeros_like(x_line)
    for i, x in enumerate(x_line):
        # 决策规则: 两个超平面的某种组合
        h1 = x + x_line[i]  # x₁ + x₂
        h2 = x - x_line[i]  # x₁ - x₂
        # 如果 h1 > 0 且 h2 > 0，则属于某类
        if h1 > 0 and h2 > 0:
            region_fill[i] = 1
    
    # 绘制最终边界
    ax2.set_title('Multiple Hyperplanes (MLP): Non-linear Boundary', 
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel(r'$x_1$', fontsize=12)
    ax2.set_ylabel(r'$x_2$', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 添加说明
    explanation = """Two hidden neurons
create non-linear
decision boundary!"""
    ax2.text(-1.8, 1.5, explanation, fontsize=10, va='center', ha='left',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    return fig

# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("线性分类与异或问题可视化")
    print("Linear Classification & XOR Problem Visualization")
    print("=" * 60)
    print()
    print("正在生成图片...")
    print()
    
    # 1. 生成线性可分图
    print("1/5: 生成线性可分问题图...")
    fig1 = plot_linear_separable()
    fig1.savefig('linear_separable.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✓ 已保存: linear_separable.png")
    plt.close(fig1)
    
    # 2. 生成线性不可分图
    print("2/5: 生成线性不可分问题图...")
    fig2 = plot_linear_inseparable()
    fig2.savefig('linear_inseparable.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✓ 已保存: linear_inseparable.png")
    plt.close(fig2)
    
    # 3. 生成异或问题图
    print("3/5: 生成异或问题图...")
    fig3 = plot_xor_problem()
    fig3.savefig('xor_problem.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✓ 已保存: xor_problem.png")
    plt.close(fig3)
    
    # 4. 生成异或问题解决方案图
    print("4/5: 生成异或问题解决方案图...")
    fig4 = plot_xor_solution()
    fig4.savefig('xor_solution.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✓ 已保存: xor_solution.png")
    plt.close(fig4)
    
    # 5. 生成组合图
    print("5/5: 生成组合对比图...")
    fig5 = create_combined_figure()
    fig5.savefig('linear_classification_combined.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✓ 已保存: linear_classification_combined.png")
    plt.close(fig5)
    
    # 6. 生成2D超平面示意图
    print("6/5: 生成2D超平面分类示意图...")
    fig6 = create_2d_hyperplane_diagram()
    fig6.savefig('hyperplane_2d.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✓ 已保存: hyperplane_2d.png")
    plt.close(fig6)
    
    print()
    print("=" * 60)
    print("所有图片生成完成!")
    print("生成的文件:")
    print("  1. linear_separable.png")
    print("  2. linear_inseparable.png")
    print("  3. xor_problem.png")
    print("  4. xor_solution.png")
    print("  5. linear_classification_combined.png")
    print("  6. hyperplane_2d.png")
    print("=" * 60)
