"""
梯度下降过程可视化生成器
Gradient Descent Visualization Generator

用于《大模型中的数学》第一章第3节微积分与优化基础
展示梯度下降算法在3D曲面、2D等高线和损失曲线上的动态过程

作者: MiniMax Agent
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# ============================================================================
# 全局配置参数
# ============================================================================

# 梯度下降超参数
LEARNING_RATE = 0.1       # 学习率
ITERATIONS = 40           # 迭代次数
START_POINT = np.array([-4.0, 3.5])  # 起始点 (x, y)

# 可视化配置
SHOW_GRID = True          # 显示网格
SAVE_GIF = True           # 是否保存GIF动画
SAVE_PNG = True           # 是否保存静态PNG
DPI = 300                 # PNG分辨率
GIF_FPS = 8               # GIF帧率
ANIMATION_INTERVAL = 150  # 动画间隔(ms)

# ============================================================================
# 数学函数定义
# ============================================================================

def loss_function(x, y):
    """
    损失函数: f(x, y) = x² + 2y²
    
    选择这个椭圆抛物面而非标准圆形的理由：
    1. 椭圆面能更好地展示梯度下降在不同方向上的收敛速度差异
    2. y方向的曲率更大(系数2)，收敛更慢，增加观察梯度修正过程的机会
    3. 更接近实际深度学习中非均匀曲率的情况
    """
    return x**2 + 2*y**2

def gradient(point):
    """
    计算梯度向量 ∇f = [∂f/∂x, ∂f/∂y]
    
    解析解:
    ∂f/∂x = 2x
    ∂f/∂y = 4y
    """
    x, y = point
    return np.array([2*x, 4*y])

def gradient_descent(start, lr, iters):
    """
    执行梯度下降算法并记录完整路径
    
    参数:
        start: 起始点坐标 (x, y)
        lr: 学习率 η
        iters: 迭代次数
    
    返回:
        path: 所有中间点的坐标，shape=(iters+1, 2)
        losses: 所有对应的损失值，shape=(iters+1,)
    """
    path = [start]
    current = start
    losses = [loss_function(current[0], current[1])]
    
    for i in range(iters):
        grad = gradient(current)
        # 梯度下降更新公式: w_{t+1} = w_t - η∇f(w_t)
        current = current - lr * grad
        path.append(current)
        losses.append(loss_function(current[0], current[1]))
    
    return np.array(path), np.array(losses)

# ============================================================================
# 可视化函数
# ============================================================================

def setup_plot_style():
    """配置全局绘图风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 设置全局字体
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10

def create_gradient_descent_animation():
    """
    创建梯度下降过程的可视化动画
    
    返回:
        fig: matplotlib.figure对象
        anim: matplotlib.animation对象
    """
    
    # 1. 执行梯度下降，获取路径数据
    path, losses = gradient_descent(START_POINT, LEARNING_RATE, ITERATIONS)
    
    # 2. 生成网格数据用于绘制背景曲面和等高线
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_function(X, Y)
    
    # 3. 创建画布 (1行3列)
    fig = plt.figure(figsize=(16, 5.5))
    fig.suptitle(
        f'Gradient Descent Optimization Process\n'
        f'Loss Function: f(x,y) = x² + 2y²  |  '
        f'Learning Rate: η = {LEARNING_RATE}  |  '
        f'Start: ({START_POINT[0]}, {START_POINT[1]})',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    # =========================================================================
    # 子图1: 3D 损失函数曲面图
    # =========================================================================
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('3D Loss Surface', fontweight='bold', pad=10)
    ax1.set_xlabel(r'$\theta_1$', fontsize=11)
    ax1.set_ylabel(r'$\theta_2$', fontsize=11)
    ax1.set_zlabel('Loss $J(\theta)$', fontsize=11)
    
    # 绘制3D曲面
    surf = ax1.plot_surface(
        X, Y, Z, 
        cmap=cm.viridis, 
        alpha=0.65, 
        edgecolor='none',
        antialiased=True
    )
    
    # 初始化3D路径线和点
    line3d, = ax1.plot([], [], [], 'r-', linewidth=2.5, 
                       marker='o', markersize=4, 
                       markerfacecolor='red', markeredgecolor='white',
                       zorder=10, label='Gradient Descent Path')
    point3d, = ax1.plot([], [], [], 'ro', markersize=8, zorder=11)
    
    # 标记起点和终点
    ax1.plot([path[0,0]], [path[0,1]], [losses[0]], 
             'go', markersize=10, markeredgecolor='white', 
             markeredgewidth=1.5, zorder=12, label='Start')
    ax1.plot([path[-1,0]], [path[-1,1]], [losses[-1]], 
             'b*', markersize=15, zorder=12, label='End')
    
    ax1.legend(loc='upper left', fontsize=9)
    ax1.view_init(elev=30, azim=-60)  # 设置观察视角
    
    # =========================================================================
    # 子图2: 2D 等高线俯视图 (Top View)
    # =========================================================================
    ax2 = fig.add_subplot(132)
    ax2.set_title('Optimization Path (Top View)', fontweight='bold', pad=10)
    ax2.set_xlabel(r'$\theta_1$', fontsize=11)
    ax2.set_ylabel(r'$\theta_2$', fontsize=11)
    ax2.set_aspect('equal')
    
    # 绘制等高线
    contour_levels = np.linspace(0, np.max(Z), 25)
    contour = ax2.contour(X, Y, Z, levels=contour_levels, 
                          cmap='viridis', alpha=0.8)
    ax2.clabel(contour, inline=True, fontsize=7, fmt='%.0f')
    
    # 绘制梯度向量场 (每隔一定间距显示)
    x_grid = np.linspace(-4, 4, 10)
    y_grid = np.linspace(-4, 4, 10)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    Z_grid = loss_function(X_grid, Y_grid)
    U = -gradient(np.array([X_grid, Y_grid]))[0]  # 负梯度方向
    V = -gradient(np.array([X_grid, Y_grid]))[1]
    
    # 归一化向量长度以便显示
    magnitude = np.sqrt(U**2 + V**2)
    U_norm = U / (magnitude + 1e-8)
    V_norm = V / (magnitude + 1e-8)
    
    ax2.quiver(X_grid, Y_grid, U_norm, V_norm, 
               magnitude, cmap='Blues', alpha=0.4, 
               scale=25, width=0.003)
    
    # 初始化2D路径
    line2d, = ax2.plot([], [], 'r-', linewidth=2, 
                       marker='o', markersize=3,
                       markerfacecolor='red', markeredgecolor='white',
                       alpha=0.8, label='Descent Path')
    point2d, = ax2.plot([], [], 'ro', markersize=8)
    
    # 标记全局最小点
    ax2.plot(0, 0, 'b*', markersize=15, 
             markeredgecolor='white', markeredgewidth=1,
             label='Global Minimum (0, 0)')
    ax2.plot(path[0,0], path[0,1], 'go', markersize=10,
             markeredgecolor='white', markeredgewidth=1.5, label='Start')
    
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    
    # 添加颜色条
    cbar = plt.colorbar(contour, ax=ax2, shrink=0.8)
    cbar.set_label('Loss Value', fontsize=10)
    
    # =========================================================================
    # 子图3: 损失值随迭代次数变化曲线
    # =========================================================================
    ax3 = fig.add_subplot(133)
    ax3.set_title('Loss vs. Iterations', fontweight='bold', pad=10)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('Loss Value $J(\theta)$', fontsize=11)
    ax3.set_xlim(-1, ITERATIONS + 2)
    ax3.set_ylim(0, max(losses) * 1.15)
    
    # 绘制理论最优损失线
    ax3.axhline(y=0, color='blue', linestyle='--', alpha=0.5, 
                label='Optimal Loss (0)')
    
    # 初始化损失曲线
    line_loss, = ax3.plot([], [], 'b-', linewidth=2.5, label='Loss Curve')
    point_loss, = ax3.plot([], [], 'ro', markersize=6)
    
    # 添加损失值文本标签
    text_info = ax3.text(
        0.02, 0.95, '', transform=ax3.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # 标记起点
    ax3.plot(0, losses[0], 'go', markersize=8, 
             markeredgecolor='white', markeredgewidth=1, label='Start')
    ax3.plot(ITERATIONS, losses[-1], 'b*', markersize=12,
             label='End')
    
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # 动画更新函数
    # =========================================================================
    
    def init():
        """初始化动画"""
        line3d.set_data([], [])
        line3d.set_3d_properties([])
        point3d.set_data([], [])
        point3d.set_3d_properties([])
        
        line2d.set_data([], [])
        point2d.set_data([], [])
        
        line_loss.set_data([], [])
        point_loss.set_data([], [])
        text_info.set_text('')
        
        return line3d, point3d, line2d, point2d, line_loss, point_loss, text_info
    
    def update(frame):
        """更新每一帧的动画内容"""
        # 获取当前帧之前的路径数据
        current_path = path[:frame+1]
        current_loss = losses[:frame+1]
        
        # 提取坐标
        x_data = current_path[:, 0]
        y_data = current_path[:, 1]
        z_data = current_loss
        
        # 更新子图1 (3D曲面)
        line3d.set_data(x_data, y_data)
        line3d.set_3d_properties(z_data)
        point3d.set_data([x_data[-1]], [y_data[-1]])
        point3d.set_3d_properties([z_data[-1]])
        
        # 更新子图2 (2D等高线)
        line2d.set_data(x_data, y_data)
        point2d.set_data([x_data[-1]], [y_data[-1]])
        
        # 更新子图3 (损失曲线)
        iterations = np.arange(len(current_loss))
        line_loss.set_data(iterations, current_loss)
        point_loss.set_data([iterations[-1]], [current_loss[-1]])
        
        # 更新信息文本
        grad_norm = np.linalg.norm(gradient(current_path[-1]))
        text_info.set_text(
            f'Iteration: {frame}\n'
            f'Loss: {current_loss[-1]:.4f}\n'
            f'||∇f||: {grad_norm:.4f}\n'
            f'Position: ({x_data[-1]:.2f}, {y_data[-1]:.2f})'
        )
        
        return line3d, point3d, line2d, point2d, line_loss, point_loss, text_info
    
    # 创建动画
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=ITERATIONS + 1,
        init_func=init,
        blit=False,
        interval=ANIMATION_INTERVAL,
        repeat=False
    )
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    return fig, anim, path, losses

def create_learning_rate_comparison():
    """
    创建不同学习率的对比可视化
    
    展示学习率对梯度下降收敛的影响:
    - 过小: 收敛太慢
    - 合适: 快速收敛
    - 过大: 震荡/发散
    """
    
    # 定义不同的学习率
    learning_rates = [0.02, 0.1, 0.3, 0.6]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Learning Rate Effects on Gradient Descent Convergence', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_function(X, Y)
    
    # 生成等高线用于所有子图
    contour_levels = np.linspace(0, 50, 20)
    
    for idx, (lr, color) in enumerate(zip(learning_rates, colors)):
        ax = axes[idx // 2, idx % 2]
        
        # 绘制等高线
        contour = ax.contour(X, Y, Z, levels=contour_levels, 
                            cmap='viridis', alpha=0.7)
        
        # 执行梯度下降
        path, losses = gradient_descent(START_POINT, lr, 30)
        
        # 绘制路径
        ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=1.5, alpha=0.7)
        ax.plot(path[:, 0], path[:, 1], 'ro', markersize=2, alpha=0.7)
        
        # 标记起点和终点
        ax.plot(path[0, 0], path[0, 1], 'go', markersize=10, 
               markeredgecolor='white', markeredgewidth=1.5, label='Start')
        ax.plot(path[-1, 0], path[-1, 1], 'b*', markersize=12,
               label='End')
        
        # 标记全局最小点
        ax.plot(0, 0, 'k*', markersize=10, alpha=0.5, label='Global Min')
        
        # 设置标题和标签
        if lr < 0.05:
            title = f'Learning Rate η = {lr} (Too Small)'
            status = 'Slow Convergence'
        elif lr < 0.2:
            title = f'Learning Rate η = {lr} (Appropriate)'
            status = 'Good Convergence ✓'
        elif lr < 0.45:
            title = f'Learning Rate η = {lr} (Large)'
            status = 'Oscillation Warning ⚠'
        else:
            title = f'Learning Rate η = {lr} (Too Large)'
            status = 'Divergence / Explosion ❌'
        
        ax.set_title(f'{title}\n{status}', fontsize=11)
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def create_loss_curve_comparison():
    """
    创建损失曲线对比图
    对比不同学习率下的损失下降曲线
    """
    
    learning_rates = [0.02, 0.1, 0.3, 0.6]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for lr, color in zip(learning_rates, colors):
        path, losses = gradient_descent(START_POINT, lr, 30)
        iterations = np.arange(len(losses))
        
        ax.plot(iterations, losses, '-', color=color, linewidth=2,
               label=f'η = {lr}', marker='o', markersize=3, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss Value $J(\theta)$', fontsize=12)
    ax.set_title('Loss Curves for Different Learning Rates', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    
    # 添加注释
    ax.annotate('Appropriate', xy=(20, 2), fontsize=10, color='#3498db')
    ax.annotate('Slow', xy=(20, 15), fontsize=10, color='#2ecc71')
    ax.annotate('Oscillation', xy=(10, 8), fontsize=10, color='#e74c3c')
    ax.annotate('Divergence', xy=(5, 20), fontsize=10, color='#9b59b6')
    
    plt.tight_layout()
    return fig

# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("梯度下降过程可视化生成器")
    print("Gradient Descent Visualization Generator")
    print("=" * 60)
    
    # 设置绘图风格
    setup_plot_style()
    
    # 1. 生成主动画
    print("\n[1/3] 生成主动画...")
    fig1, anim, path, losses = create_gradient_descent_animation()
    
    if SAVE_GIF:
        try:
            print("    保存GIF动画...")
            anim.save('gradient_descent_animation.gif', 
                     writer='pillow', fps=GIF_FPS, dpi=100)
            print("    ✓ 已保存: gradient_descent_animation.gif")
        except Exception as e:
            print(f"    ✗ GIF保存失败: {e}")
            print("    (可能需要安装 pillow: pip install pillow)")
    
    if SAVE_PNG:
        print("    生成静态高清图...")
        update(ITERATIONS)
        fig1.savefig('gradient_descent_final.png', 
                    dpi=DPI, bbox_inches='tight', facecolor='white')
        print("    ✓ 已保存: gradient_descent_final.png")
    
    plt.close(fig1)
    
    # 2. 生成学习率对比图
    print("\n[2/3] 生成学习率对比图...")
    fig2 = create_learning_rate_comparison()
    fig2.savefig('learning_rate_comparison.png', 
                dpi=DPI, bbox_inches='tight', facecolor='white')
    print("    ✓ 已保存: learning_rate_comparison.png")
    plt.close(fig2)
    
    # 3. 生成损失曲线对比图
    print("\n[3/3] 生成损失曲线对比图...")
    fig3 = create_loss_curve_comparison()
    fig3.savefig('loss_curves_comparison.png', 
                dpi=DPI, bbox_inches='tight', facecolor='white')
    print("    ✓ 已保存: loss_curves_comparison.png")
    plt.close(fig3)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("优化统计信息:")
    print(f"  起始点: ({START_POINT[0]}, {START_POINT[1]})")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  迭代次数: {ITERATIONS}")
    print(f"  起始损失: {losses[0]:.4f}")
    print(f"  最终损失: {losses[-1]:.6f}")
    print(f"  最终位置: ({path[-1,0]:.6f}, {path[-1,1]:.6f})")
    print(f"  梯度范数: {np.linalg.norm(gradient(path[-1])):.6f}")
    print("=" * 60)
    print("\n生成完成！所有文件已保存到当前目录。")
