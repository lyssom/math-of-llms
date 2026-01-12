"""
梯度下降过程可视化 - Manim版本
Gradient Descent Visualization using Manim

用于《大模型中的数学》第一章第3节微积分与优化基础
生成高质量数学动画，展示梯度下降算法在3D曲面和2D等高线上的动态过程

作者: MiniMax Agent

使用说明:
1. 安装依赖: pip install manim
2. 运行: manim -pql gradient_descent_manim.py GradientDescentScene
3. 参数说明: -p (预览), -q (质量: l/m/h/k), -l (低质量,快速)
"""

from manim import *
import numpy as np

# ============================================================================
# 配置参数
# ============================================================================

# 梯度下降超参数
LEARNING_RATE = 0.1
ITERATIONS = 40
START_POINT = np.array([-3.5, 2.5])  # 起始点 (x, y)

# 动画配置
ANIMATION_SPEED = 0.8
FRAME_RATE = 30

# ============================================================================
# 数学函数定义
# ============================================================================

def loss_function(x, y):
    """
    损失函数: f(x, y) = x² + 2y²
    使用椭圆抛物面而非标准圆形，以展示不同方向的收敛速度差异
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
    """
    path = [start]
    current = start
    losses = [loss_function(current[0], current[1])]
    
    for i in range(iters):
        grad = gradient(current)
        current = current - lr * grad
        path.append(current)
        losses.append(loss_function(current[0], current[1]))
    
    return np.array(path), np.array(losses)

# ============================================================================
# 场景类定义
# ============================================================================

class GradientDescentScene(Scene):
    """
    梯度下降过程可视化场景
    
    展示:
    1. 3D曲面视图 (透视)
    2. 2D等高线视图 (俯视)
    3. 损失曲线下降
    4. 动态优化路径
    """
    
    def construct(self):
        """主场景构建"""
        
        # 1. 准备数据
        path, losses = gradient_descent(START_POINT, LEARNING_RATE, ITERATIONS)
        
        # 生成网格数据
        x_range = np.linspace(-4, 4, 50)
        y_range = np.linspace(-4, 4, 50)
        X, Y = np.meshgrid(x_range, y_range)
        Z = loss_function(X, Y)
        
        # =========================================================================
        # 场景1: 等高线图和优化路径
        # =========================================================================
        
        # 创建标题
        title = Text(
            f'Gradient Descent Visualization\n'
            f'Loss: $f(x,y) = x^2 + 2y^2$, Learning Rate: $\\eta = {LEARNING_RATE}$',
            font_size=24
        )
        title.to_edge(UP, buff=0.5)
        self.add(title)
        
        # 创建等高线图
        print("正在生成等高线数据...")
        contour_group = self.create_contour_plot(X, Y, Z)
        contour_group.move_to(LEFT * 2)
        self.add(contour_group)
        
        # 绘制优化路径
        path_line = self.create_path_line(path)
        self.add(path_line)
        
        # 创建移动点
        current_point = Dot(point=path_line.get_start(), radius=0.1, color=RED)
        self.add(current_point)
        
        # 创建梯度向量箭头
        gradient_arrow = self.create_gradient_arrow(path[0])
        self.add(gradient_arrow)
        
        # =========================================================================
        # 场景2: 损失曲线
        # =========================================================================
        
        # 创建损失曲线图
        loss_graph = self.create_loss_graph(losses, ITERATIONS)
        loss_graph.move_to(RIGHT * 2 + DOWN * 0.5)
        self.add(loss_graph)
        
        # 创建损失曲线
        loss_line = self.create_loss_line(losses, ITERATIONS, loss_graph)
        self.add(loss_line)
        
        # =========================================================================
        # 动画: 梯度下降过程
        # =========================================================================
        
        print("正在生成动画...")
        
        # 创建坐标轴标签
        x_label = MathTex(r'\theta_1').next_to(contour_group, DOWN, buff=0.3)
        y_label = MathTex(r'\theta_2').next_to(contour_group, LEFT, buff=0.3)
        self.add(x_label, y_label)
        
        # 动画：点沿着路径移动
        for i in range(len(path)):
            new_point = Dot(
                point=path_line.get_points()[:, i], 
                radius=0.08, 
                color=RED
            )
            
            # 更新路径颜色（走过的路径变暗）
            if i > 0:
                path_line.set_color_by_index(
                    range(i * 3, len(path_line.get_points()[0])), 
                    RED_E
                )
            
            # 移动点
            self.play(
                Transform(current_point, new_point),
                run_time=0.1,
                rate_func=linear
            )
            
            # 更新梯度箭头
            if i < len(path) - 1:
                new_arrow = self.create_gradient_arrow(path[i+1])
                self.play(
                    Transform(gradient_arrow, new_arrow),
                    run_time=0.05
                )
            
            # 绘制损失曲线
            if i > 0 and i % 2 == 0:
                partial_line = self.create_partial_loss_line(
                    losses[:i+1], i, loss_graph
                )
                self.add(partial_line)
        
        # 最终标注
        final_loss = MathTex(
            f'Final\\ Loss: {losses[-1]:.4f}',
            font_size=18,
            color=BLUE
        )
        final_loss.next_to(loss_graph, UP, buff=0.3)
        self.play(Write(final_loss))
        
        # 等待一段时间展示最终效果
        self.wait(2)
        
        print("动画生成完成!")
    
    def create_contour_plot(self, X, Y, Z):
        """创建等高线图"""
        
        # 手动创建等高线（因为Manim的Contour需要特殊处理）
        contour_lines = VGroup()
        
        # 创建不同高度的水平线来模拟等高线
        levels = [1, 2, 4, 6, 8, 10, 15, 20, 30]
        
        for level in levels:
            # 计算等高线的点
            points = []
            for y in np.linspace(-4, 4, 100):
                # x² + 2y² = level => x = ±√(level - 2y²)
                if level - 2*y**2 >= 0:
                    x = np.sqrt(level - 2*y**2)
                    points.append([x, y, 0])
                    points.append([-x, y, 0])
            
            if len(points) > 1:
                points = np.array(points)
                line = VMobject()
                line.set_points_smoothly(points[:, :2])
                line.set_stroke(
                    color=self.get_color_by_value(level, 1, 30),
                    width=1.5,
                    opacity=0.8
                )
                contour_lines.add(line)
        
        # 创建边界框
        bounding_box = Rectangle(
            width=6, height=6,
            color=WHITE,
            fill_opacity=0,
            stroke_width=2
        )
        
        # 组合
        group = VGroup(contour_lines, bounding_box)
        group.set_width(5)
        group.set_height(5)
        
        return group
    
    def get_color_by_value(self, value, min_val, max_val):
        """根据值获取颜色（从蓝到红）"""
        t = (value - min_val) / (max_val - min_val)
        if t < 0.33:
            return BLUE
        elif t < 0.66:
            return YELLOW
        else:
            return RED
    
    def create_path_line(self, path):
        """创建优化路径线"""
        # 将路径点转换为3D点
        points_3d = []
        for p in path:
            z = loss_function(p[0], p[1])
            points_3d.append([p[0], p[1], z])
        points_3d = np.array(points_3d)
        
        # 归一化到显示坐标
        scale = 0.5
        normalized_points = points_3d[:, :2] * scale
        
        line = VMobject()
        line.set_points_smoothly(normalized_points)
        line.set_stroke(color=RED, width=3)
        
        return line
    
    def create_gradient_arrow(self, point):
        """创建梯度向量箭头"""
        grad = gradient(point)
        
        # 计算箭头终点（负梯度方向）
        end_point = point - LEARNING_RATE * grad
        
        # 归一化到显示坐标
        scale = 0.5
        start = point[:2] * scale
        end = end_point[:2] * scale
        
        # 创建箭头
        arrow = Arrow(
            start=start,
            end=end,
            color=YELLOW,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.2
        )
        
        return arrow
    
    def create_loss_graph(self, losses, iterations):
        """创建损失曲线图坐标系"""
        
        # 创建坐标轴
        x_max = iterations
        y_max = max(losses) * 1.1
        
        # 创建坐标轴
        x_axis = NumberLine(
            x_range=[0, x_max, 10],
            length=4,
            color=WHITE
        )
        x_axis.rotate(PI / 2)
        x_axis.move_to(UP * 2 + LEFT * 3)
        
        y_axis = NumberLine(
            x_range=[0, y_max, 5],
            length=3,
            color=WHITE
        )
        y_axis.move_to(UP * 0.5 + LEFT * 3)
        
        # 创建坐标框
        graph_box = VGroup(x_axis, y_axis)
        
        return graph_box
    
    def create_loss_line(self, losses, iterations, graph):
        """创建损失曲线"""
        
        # 转换损失值为坐标
        y_max = max(losses) * 1.1
        scale_y = 3 / y_max
        scale_x = 4 / iterations
        
        points = []
        for i, loss in enumerate(losses):
            x = -1.5 + i * scale_x  # 从左侧开始
            y = -1 + loss * scale_y
            points.append([x, y, 0])
        
        points = np.array(points)
        
        line = VMobject()
        line.set_points_smoothly(points[:, :2])
        line.set_stroke(color=BLUE, width=2)
        
        return line
    
    def create_partial_loss_line(self, losses, current_iter, graph):
        """创建部分损失曲线（用于动画）"""
        
        y_max = max(losses) * 1.1
        scale_y = 3 / y_max
        scale_x = 4 / len(losses)
        
        points = []
        for i, loss in enumerate(losses):
            x = -1.5 + i * scale_x
            y = -1 + loss * scale_y
            points.append([x, y, 0])
        
        points = np.array(points)
        
        line = VMobject()
        line.set_points_smoothly(points[:, :2])
        line.set_stroke(color=BLUE, width=2)
        
        return line


class GradientDescent3DScene(ThreeDScene):
    """
    3D梯度下降可视化场景
    
    展示:
    - 3D损失函数曲面
    - 曲面下降路径
    - 动态小球沿路径移动
    """
    
    def construct(self):
        """主场景构建"""
        
        # 设置相机
        self.set_camera_orientation(
            phi=75 * DEGREES,
            theta=30 * DEGREES
        )
        
        # 1. 准备数据
        path, losses = gradient_descent(START_POINT, LEARNING_RATE, ITERATIONS)
        
        # 生成网格数据
        x_range = np.linspace(-3, 3, 30)
        y_range = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(x_range, y_range)
        Z = loss_function(X, Y)
        
        # 2. 创建标题
        title = Text(
            'Gradient Descent in 3D\nLoss Surface Visualization',
            font_size=28
        )
        title.to_edge(UP)
        self.add(title)
        
        # 3. 创建3D曲面
        print("正在生成3D曲面...")
        surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                loss_function(u, v) / 3  # 缩放Z轴
            ]),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(30, 30),
            surface_piece_config={'.checkerboard_diffraction': False}
        )
        surface.set_style(
            fill_opacity=0.7,
            stroke_color=WHITE,
            stroke_width=0.5
        )
        surface.set_fill_by_value(
            axes=Z,
            cmap='viridis',
            max=10,
            min=0
        )
        
        self.play(
            Create(surface),
            run_time=3
        )
        
        # 4. 创建路径
        path_points = []
        for p in path:
            path_points.append([
                p[0],
                p[1],
                loss_function(p[0], p[1]) / 3
            ])
        path_points = np.array(path_points)
        
        path_line = Line3D(
            path_points[0],
            path_points[1],
            color=RED,
            stroke_width=4
        )
        
        for i in range(1, len(path_points)):
            new_segment = Line3D(
                path_points[i],
                path_points[i+1] if i < len(path_points)-1 else path_points[i],
                color=RED,
                stroke_width=4
            )
            path_line.add(new_segment)
        
        self.play(Create(path_line), run_time=2)
        
        # 5. 创建移动小球
        sphere = Sphere(
            center=path_points[0],
            radius=0.15,
            resolution=(8, 8)
        )
        sphere.set_color(RED_E)
        
        self.add(sphere)
        
        # 6. 动画：小球沿路径移动
        print("正在生成3D动画...")
        for i in range(len(path_points)):
            target_point = path_points[i]
            
            # 更新小球位置
            new_sphere = Sphere(
                center=target_point,
                radius=0.15,
                resolution=(8, 8)
            )
            new_sphere.set_color(
                RED if i > len(path_points) * 0.7 else 
                ORANGE if i > len(path_points) * 0.4 else YELLOW
            )
            
            self.play(
                Transform(sphere, new_sphere),
                run_time=0.1,
                rate_func=linear
            )
        
        # 7. 添加标签
        loss_label = MathTex(
            f'Final\\ Loss: {losses[-1]:.4f}',
            font_size=24,
            color=BLUE
        )
        loss_label.to_corner(DOWN + RIGHT)
        self.play(Write(loss_label))
        
        self.wait(2)
        print("3D场景生成完成!")


class LearningRateComparisonScene(Scene):
    """
    学习率对比可视化场景
    
    展示不同学习率对梯度下降收敛的影响
    """
    
    def construct(self):
        """主场景构建"""
        
        # 标题
        title = Text(
            'Learning Rate Effects on Convergence',
            font_size=28,
            color=BLUE
        )
        title.to_edge(UP, buff=0.5)
        self.add(title)
        
        # 定义不同的学习率
        learning_rates = [0.02, 0.1, 0.3, 0.6]
        colors = [GREEN, BLUE, ORANGE, RED]
        labels = ['Too Small', 'Appropriate', 'Large', 'Too Large']
        
        # 创建四个子图
        positions = [
            (UP + LEFT),
            (UP + RIGHT),
            (DOWN + LEFT),
            (DOWN + RIGHT)
        ]
        
        for lr, color, label, pos in zip(learning_rates, colors, labels, positions):
            # 创建子图
            subgraph = self.create_subplot(lr, color, label)
            subgraph.move_to(pos * 0.5)
            self.add(subgraph)
        
        self.wait(2)
    
    def create_subplot(self, lr, color, label):
        """创建单个学习率子图"""
        
        # 执行梯度下降
        path, losses = gradient_descent(START_POINT, lr, 20)
        
        # 创建坐标框
        box = Rectangle(
            width=3, height=2.5,
            color=WHITE,
            fill_opacity=0.1,
            stroke_width=1
        )
        
        # 创建标题
        title = Text(
            f'η = {lr}\n({label})',
            font_size=14,
            color=color
        )
        title.next_to(box, UP, buff=0.1)
        
        # 创建损失曲线
        y_max = max(losses) * 1.1
        points = []
        for i, loss in enumerate(losses):
            x = -1.2 + i * (2.4 / len(losses))
            y = -1 + loss * (2 / y_max)
            points.append([x, y, 0])
        points = np.array(points)
        
        line = VMobject()
        line.set_points_smoothly(points[:, :2])
        line.set_stroke(color=color, width=2)
        
        # 最终损失值
        final_loss = MathTex(
            f'{losses[-1]:.1f}',
            font_size=12,
            color=color
        )
        final_loss.next_to(line.get_end(), UP, buff=0.1)
        
        group = VGroup(box, title, line, final_loss)
        return group


# ============================================================================
# 辅助函数
# ============================================================================

def create_3d_surface_plot():
    """
    创建静态3D曲面图（用于导出为PNG）
    """
    from manim import ThreeDScene
    import numpy as np
    
    # 创建场景
    config.pixel_height = 1080
    config.pixel_width = 1920
    config.frame_rate = 30
    
    # 准备数据
    path, losses = gradient_descent(START_POINT, LEARNING_RATE, ITERATIONS)
    
    x_range = np.linspace(-3, 3, 40)
    y_range = np.linspace(-3, 3, 40)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_function(X, Y)
    
    # 创建曲面
    surface = Surface(
        lambda u, v: np.array([
            u,
            v,
            loss_function(u, v) / 3
        ]),
        u_range=[-3, 3],
        v_range=[-3, 3],
        resolution=(40, 40)
    )
    surface.set_style(
        fill_opacity=0.8,
        stroke_color=WHITE,
        stroke_width=0.5
    )
    surface.set_fill_by_value(
        axes=Z,
        cmap='viridis',
        max=10,
        min=0
    )
    
    return surface


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("梯度下降可视化 - Manim版本")
    print("Gradient Descent Visualization (Manim)")
    print("=" * 60)
    print()
    print("使用说明:")
    print("  1. 确保已安装 manim: pip install manim")
    print("  2. 运行主场景:")
    print("     manim -pql gradient_descent_manim.py GradientDescentScene")
    print("  3. 运行3D场景:")
    print("     manim -pql gradient_descent_manim.py GradientDescent3DScene")
    print("  4. 运行学习率对比:")
    print("     manim -pql gradient_descent_manim.py LearningRateComparisonScene")
    print()
    print("参数说明:")
    print("  -p: 预览")
    print("  -q: 质量 (l=低, m=中, h=高, k=4K)")
    print("  -l: 低质量快速预览")
    print()
    print("示例:")
    print("  manim -pl gradient_descent_manim.py GradientDescentScene")
    print()
    print("=" * 60)
    
    # 打印优化信息
    path, losses = gradient_descent(START_POINT, LEARNING_RATE, ITERATIONS)
    print("\n优化统计:")
    print(f"  起始点: ({START_POINT[0]}, {START_POINT[1]})")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  迭代次数: {ITERATIONS}")
    print(f"  起始损失: {losses[0]:.4f}")
    print(f"  最终损失: {losses[-1]:.6f}")
    print(f"  收敛比例: {(1 - losses[-1]/losses[0])*100:.2f}%")
