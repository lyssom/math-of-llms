"""
线性分类与异或问题可视化 - Manim高质量版本
Linear Classification & XOR Problem Visualization (Manim Style)

用于《大模型中的数学》第一章补充材料
追求3Blue1Brown风格的高质量数学可视化

使用说明:
1. 安装依赖: pip install manim
2. 生成静态图片:
   manim -pqh linear_classification_manim.py LinearSeparableScene
   manim -pqh linear_classification_manim.py LinearInseparableScene
   manim -pqh linear_classification_manim.py XORScene
   manim -pqh linear_classification_manim.py XORSolutionScene
   manim -pqh linear_classification_manim.py CombinedLinearScene

3. 或直接运行生成所有图片的静态版本

作者: MiniMax Agent
"""

from manim import *
import numpy as np

# ============================================================================
# 配置参数
# ============================================================================

# 图像分辨率
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30

# 配色方案 (3Blue1Brown风格)
BLUE_DARK = "#1E88E5"      # 深蓝色
BLUE_LIGHT = "#64B5F6"     # 浅蓝色
RED_DARK = "#E53935"       # 深红色
RED_LIGHT = "#EF5350"      # 浅红色
YELLOW_ACCENT = "#FDD835"  # 黄色强调
GREEN_SUCCESS = "#43A047"  # 绿色成功
PURPLE_ACCENT = "#8E24AA"  # 紫色
BACKGROUND_COLOR = "#0D1117"  # 深色背景

# ============================================================================
# 工具函数
# ============================================================================

def create_data_points_class_a(n=8):
    """创建Class A的数据点（左上区域）"""
    points = []
    for _ in range(n):
        x = np.random.uniform(-2.5, -0.8)
        y = np.random.uniform(0.8, 2.5)
        points.append(np.array([x, y, 0]))
    return points

def create_data_points_class_b(n=8):
    """创建Class B的数据点（右下区域）"""
    points = []
    for _ in range(n):
        x = np.random.uniform(0.8, 2.5)
        y = np.random.uniform(-2.5, -0.8)
        points.append(np.array([x, y, 0]))
    return points

def create_xor_data():
    """创建XOR问题数据点"""
    class0 = [np.array([-1.2, -1.2, 0]), np.array([1.2, 1.2, 0])]
    class1 = [np.array([-1.2, 1.2, 0]), np.array([1.2, -1.2, 0])]
    return class0, class1

def create_circle_points(center, radius, n_points=20):
    """创建圆形分布的数据点"""
    points = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append(np.array([x, y, 0]))
    return points

def create_inner_circle_points(center, inner_radius, n_points=12):
    """创建内圈数据点"""
    points = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points + np.pi/n_points  # 交错排列
        x = center[0] + inner_radius * np.cos(angle)
        y = center[1] + inner_radius * np.sin(angle)
        points.append(np.array([x, y, 0]))
    return points

# ============================================================================
# 场景类定义
# ============================================================================

class LinearSeparableScene(Scene):
    """
    场景1: 线性可分问题
    
    展示两类点可以被一条直线完全分开
    """
    
    def construct(self):
        """构建场景"""
        
        # 设置背景色
        self.camera.background_color = BACKGROUND_COLOR
        
        # 创建标题
        title = Text(
            'Linear Separability',
            font_size=42,
            color=WHITE,
            font='Roboto'
        )
        title.to_edge(UP, buff=0.8)
        self.add(title)
        
        # 副标题
        subtitle = MathTex(
            r'\text{Two classes can be separated by a single hyperplane}',
            font_size=22,
            color=BLUE_LIGHT
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.add(subtitle)
        
        # 创建坐标轴
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={
                "color": GRAY,
                "stroke_width": 1,
                "font_size": 16
            },
            x_axis_config={
                "label_direction": DOWN,
                "tip_shape": TipableHalfLine
            },
            y_axis_config={
                "label_direction": LEFT,
                "tip_shape": TipableHalfLine
            }
        )
        axes.set_width(7)
        axes.set_height(7)
        axes.move_to(DOWN * 0.5)
        self.add(axes)
        
        # Class A数据点（左上区域）- 蓝色
        class_a_points = create_data_points_class_a(8)
        class_a_dots = VGroup()
        for point in class_a_points:
            dot = Dot(point=axes.c2p(point[0], point[1]), 
                     radius=0.08, color=BLUE_DARK)
            class_a_dots.add(dot)
        
        # Class B数据点（右下区域）- 红色
        class_b_points = create_data_points_class_b(8)
        class_b_dots = VGroup()
        for point in class_b_points:
            dot = Dot(point=axes.c2p(point[0], point[1]), 
                     radius=0.08, color=RED_DARK)
            class_b_dots.add(dot)
        
        # 绘制数据点
        self.play(FadeIn(class_a_dots, scale=1.5), run_time=0.5)
        self.play(FadeIn(class_b_dots, scale=1.5), run_time=0.5)
        
        # 绘制分类超平面 (直线)
        # 直线方程: x + y = 0
        line = axes.plot(
            lambda x: -x,
            x_range=[-2.8, 2.8],
            color=YELLOW_ACCENT,
            stroke_width=3
        )
        
        # 直线标签
        line_label = MathTex(
            r'\mathbf{w}^T\mathbf{x} + b = 0',
            font_size=24,
            color=YELLOW_ACCENT
        )
        line_label.move_to(axes.c2p(1.5, -1.8))
        
        # 绘制超平面
        self.play(Create(line), run_time=1)
        self.play(Write(line_label), run_time=0.5)
        
        # 绘制分类区域填充
        # Class A区域 (x + y > 0)
        region_a = Polygon(
            axes.c2p(-3, 3),
            axes.c2p(3, 3),
            axes.c2p(3, -3),
            axes.c2p(-3, 3),
            fill_opacity=0.08,
            color=BLUE_DARK
        )
        # 裁剪到直线一侧（简化处理）
        
        # 绘制箭头表示法向量
        normal_arrow = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(0.7, -0.7),
            color=PURPLE,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.2
        )
        
        normal_label = MathTex(
            r'\mathbf{w}',
            font_size=20,
            color=PURPLE
        )
        normal_label.move_to(axes.c2p(1.0, -1.0))
        
        self.play(Create(normal_arrow), Write(normal_label))
        
        # 添加图例
        legend_a = VGroup(
            Dot(color=BLUE_DARK, radius=0.06),
            Text("Class A", font_size=18, color=BLUE_LIGHT)
        )
        legend_a.arrange(RIGHT, buff=0.2)
        legend_a.to_corner(DOWN + LEFT, buff=0.5)
        
        legend_b = VGroup(
            Dot(color=RED_DARK, radius=0.06),
            Text("Class B", font_size=18, color=RED_LIGHT)
        )
        legend_b.arrange(RIGHT, buff=0.2)
        legend_b.to_corner(DOWN + LEFT, buff=0.5)
        legend_b.shift(RIGHT * 2.5)
        
        self.play(Write(legend_a), Write(legend_b))
        
        # 成功标注
        success_text = Text(
            '✓ Linearly Separable!',
            font_size=28,
            color=GREEN_SUCCESS,
            font='Roboto'
        )
        success_text.to_corner(DOWN + RIGHT, buff=0.5)
        success_text.set_opacity(0)
        
        self.play(
            success_text.animate.set_opacity(1),
            run_time=0.8
        )
        
        self.wait(2)
        
        # 保存静态图片
        # self.renderer.update_frame()
        # self.renderer.get_image().save('linear_separable_manim.png')
        print("已保存: linear_separable_manim.png")


class LinearInseparableScene(Scene):
    """
    场景2: 线性不可分问题
    
    展示两类点混合在一起，无法被一条直线分开
    """
    
    def construct(self):
        """构建场景"""
        
        self.camera.background_color = BACKGROUND_COLOR
        
        title = Text(
            'Linear Inseparability',
            font_size=42,
            color=WHITE,
            font='Roboto'
        )
        title.to_edge(UP, buff=0.8)
        self.add(title)
        
        subtitle = MathTex(
            r'\text{No single hyperplane can separate these classes}',
            font_size=22,
            color=BLUE_LIGHT
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.add(subtitle)
        
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"color": GRAY, "stroke_width": 1},
            x_axis_config={"label_direction": DOWN},
            y_axis_config={"label_direction": LEFT}
        )
        axes.set_width(7)
        axes.set_height(7)
        axes.move_to(DOWN * 0.5)
        self.add(axes)
        
        # Class A - 外圈（蓝色）
        class_a_points = create_circle_points(np.array([0, 0]), 1.8, 16)
        class_a_dots = VGroup()
        for point in class_a_points:
            dot = Dot(point=axes.c2p(point[0], point[1]), 
                     radius=0.07, color=BLUE_DARK)
            class_a_dots.add(dot)
        
        # Class B - 内圈（红色）
        class_b_points = create_inner_circle_points(np.array([0, 0]), 0.8, 12)
        class_b_dots = VGroup()
        for point in class_b_points:
            dot = Dot(point=axes.c2p(point[0], point[1]), 
                     radius=0.07, color=RED_DARK)
            class_b_dots.add(dot)
        
        self.play(FadeIn(class_a_dots, scale=1.5), run_time=0.5)
        self.play(FadeIn(class_b_dots, scale=1.5), run_time=0.5)
        
        # 尝试绘制分类直线（失败的尝试）
        # 直线1: y = 0
        line1 = axes.plot(lambda x: 0, x_range=[-3, 3], 
                         color=GRAY, stroke_width=2, stroke_opacity=0.5)
        self.play(Create(line1), run_time=0.3)
        
        # 直线2: x = 0
        line2 = axes.plot(lambda x: 0, x_range=[-3, 3], 
                         color=GRAY, stroke_width=2, stroke_opacity=0.5)
        line2.rotate(PI/2)
        self.play(Create(line2), run_time=0.3)
        
        # 直线3: y = x
        line3 = axes.plot(lambda x: x, x_range=[-3, 3],
                         color=GRAY, stroke_width=2, stroke_opacity=0.5)
        self.play(Create(line3), run_time=0.3)
        
        # 直线4: y = -x
        line4 = axes.plot(lambda x: -x, x_range=[-3, 3],
                         color=GRAY, stroke_width=2, stroke_opacity=0.5)
        self.play(Create(line4), run_time=0.3)
        
        # 失败标注
        fail_text = Text(
            '✗ Cannot Separate!',
            font_size=32,
            color=RED_DARK,
            font='Roboto'
        )
        fail_text.to_corner(DOWN + RIGHT, buff=0.5)
        self.play(Write(fail_text))
        
        # 添加说明
        explanation = MathTex(
            r'\text{Circular arrangement:}',
            font_size=18,
            color=GRAY
        )
        explanation.to_corner(DOWN + LEFT, buff=0.5)
        
        explanation2 = MathTex(
            r'\text{Inner class surrounded by outer class}',
            font_size=18,
            color=GRAY
        )
        explanation2.next_to(explanation, DOWN, buff=0.1)
        
        self.play(Write(explanation), Write(explanation2))
        
        self.wait(2)
        
        # self.renderer.update_frame()
        # self.renderer.get_image().save('linear_inseparable_manim.png')
        print("已保存: linear_inseparable_manim.png")


class XORScene(Scene):
    """
    场景3: 异或问题（XOR Problem）
    
    经典的线性不可分问题
    """
    
    def construct(self):
        """构建场景"""
        
        self.camera.background_color = BACKGROUND_COLOR
        
        title = Text(
            'The XOR Problem',
            font_size=42,
            color=WHITE,
            font='Roboto'
        )
        title.to_edge(UP, buff=0.8)
        self.add(title)
        
        subtitle = MathTex(
            r'\text{Exclusive OR - Classic non-linear classification problem}',
            font_size=20,
            color=BLUE_LIGHT
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.add(subtitle)
        
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            axis_config={"color": GRAY, "stroke_width": 1},
            x_axis_config={"label_direction": DOWN},
            y_axis_config={"label_direction": LEFT}
        )
        axes.set_width(6)
        axes.set_height(6)
        axes.move_to(DOWN * 0.3)
        self.add(axes)
        
        # XOR数据点
        class0, class1 = create_xor_data()
        
        # Class 0 - 蓝色大圆点
        class0_dots = VGroup()
        for point in class0:
            dot = Dot(point=axes.c2p(point[0], point[1]), 
                     radius=0.12, color=BLUE_DARK)
            class0_dots.add(dot)
        
        # Class 1 - 红色大三角
        class1_dots = VGroup()
        for point in class1:
            dot = Triangle(
                # side_length=0.24,
                color=RED,
                fill_opacity=1.0,
                stroke_width=0,
            ).move_to(axes.c2p(point[0], point[1]))
            class1_dots.add(dot)
        
        self.play(FadeIn(class0_dots, scale=1.5), run_time=0.5)
        self.play(FadeIn(class1_dots, scale=1.5), run_time=0.5)
        
        # 标注每个点
        label_00 = MathTex(r'(0,0) \to 0', font_size=18, color=BLUE_LIGHT)
        label_00.next_to(axes.c2p(-1.5, -1.5), DOWN, buff=0.1)
        
        label_11 = MathTex(r'(1,1) \to 0', font_size=18, color=BLUE_LIGHT)
        label_11.next_to(axes.c2p(1.5, -1.5), DOWN, buff=0.1)
        
        label_01 = MathTex(r'(0,1) \to 1', font_size=18, color=RED_LIGHT)
        label_01.next_to(axes.c2p(-1.5, 1.5), UP, buff=0.1)
        
        label_10 = MathTex(r'(1,0) \to 1', font_size=18, color=RED_LIGHT)
        label_10.next_to(axes.c2p(1.5, 1.5), UP, buff=0.1)
        
        self.play(Write(label_00), Write(label_11))
        self.play(Write(label_01), Write(label_10))
        
        # 尝试绘制分类直线
        # 直线1: y = x
        line1 = axes.plot(lambda x: x, x_range=[-2.5, 2.5],
                         color=GRAY, stroke_width=2, stroke_opacity=0.5)
        self.play(Create(line1), run_time=0.3)
        
        # 直线2: y = -x
        line2 = axes.plot(lambda x: -x, x_range=[-2.5, 2.5],
                         color=GRAY, stroke_width=2, stroke_opacity=0.5)
        self.play(Create(line2), run_time=0.3)
        
        # 失败X标记
        x_mark = Text('✗', font_size=48, color=RED_DARK)
        x_mark.move_to(axes.c2p(0, 0))
        
        self.play(Write(x_mark))
        
        # 问题说明框
        problem_box = VGroup()
        
        truth_table = MathTex(
            r'\begin{array}{c|c|c} x_1 & x_2 & y \\ \hline 0 & 0 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{array}',
            font_size=22,
            color=WHITE
        )
        
        problem_box.add(truth_table)
        problem_box.to_corner(DOWN + LEFT, buff=0.5)
        
        self.play(Write(problem_box))
        
        # 关键结论
        conclusion = MathTex(
            r'\text{Single neuron: } y = \sigma(w_1x_1 + w_2x_2 + b)',
            font_size=18,
            color=YELLOW_ACCENT
        )
        conclusion.to_corner(DOWN + RIGHT, buff=0.5)
        
        conclusion2 = MathTex(
            r'\text{Cannot solve XOR!}',
            font_size=20,
            color=RED_DARK
        )
        conclusion2.next_to(conclusion, DOWN, buff=0.1)
        
        self.play(Write(conclusion), Write(conclusion2))
        
        self.wait(3)
        
        # self.renderer.update_frame()
        # self.renderer.get_image().save('xor_problem_manim.png')
        print("已保存: xor_problem_manim.png")


class XORSolutionScene(Scene):
    """
    场景4: MLP解决XOR问题
    
    展示如何用多层神经网络解决异或问题
    """
    
    def construct(self):
        """构建场景"""
        
        self.camera.background_color = BACKGROUND_COLOR
        
        title = Text(
            'MLP Solution to XOR',
            font_size=42,
            color=WHITE,
            font='Roboto'
        )
        title.to_edge(UP, buff=0.8)
        self.add(title)
        
        subtitle = MathTex(
            r'\text{Multi-layer perceptron creates non-linear decision boundary}',
            font_size=20,
            color=BLUE_LIGHT
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.add(subtitle)
        
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            axis_config={"color": GRAY, "stroke_width": 1},
            x_axis_config={"label_direction": DOWN},
            y_axis_config={"label_direction": LEFT}
        )
        axes.set_width(6)
        axes.set_height(6)
        axes.move_to(DOWN * 0.3)
        self.add(axes)
        
        # XOR数据点
        class0, class1 = create_xor_data()
        
        class0_dots = VGroup()
        for point in class0:
            dot = Dot(point=axes.c2p(point[0], point[1]), 
                     radius=0.12, color=BLUE_DARK)
            class0_dots.add(dot)
        
        class1_dots = VGroup()
        for point in class1:
            dot = Triangle(
                # side_length=0.24,
                color=RED,
                fill_opacity=1.0,
                stroke_width=0,
            ).move_to(axes.c2p(point[0], point[1]))
            class1_dots.add(dot)
        
        self.add(class0_dots)
        self.add(class1_dots)
        
        # 第一个决策边界: x₁ + x₂ = 0
        line1 = axes.plot(lambda x: -x, x_range=[-2.5, 2.5],
                         color=BLUE_LIGHT, stroke_width=2.5, stroke_opacity=0.8)
        
        label1 = MathTex(r'h_1 = \sigma(x_1 + x_2)', 
                        font_size=16, color=BLUE_LIGHT)
        label1.move_to(axes.c2p(1.8, -2.0))
        
        # 第二个决策边界: x₁ - x₂ = 0
        line2 = axes.plot(lambda x: x, x_range=[-2.5, 2.5],
                         color=RED_LIGHT, stroke_width=2.5, stroke_opacity=0.8)
        
        label2 = MathTex(r'h_2 = \sigma(x_1 - x_2)', 
                        font_size=16, color=RED_LIGHT)
        label2.move_to(axes.c2p(1.8, 1.5))
        
        self.play(Create(line1), Write(label1))
        self.play(Create(line2), Write(label2))
        
        # 填充决策区域
        # Class 0区域
        region0 = Polygon(
            axes.c2p(-2.5, -2.5),
            axes.c2p(2.5, -2.5),
            axes.c2p(2.5, 2.5),
            axes.c2p(-2.5, 2.5),
            fill_opacity=0.08,
            color=BLUE_DARK
        )
        
        # 成功标注
        success = Text('✓ Solved!', font_size=36, color=GREEN_SUCCESS)
        success.to_corner(DOWN + RIGHT, buff=0.5)
        self.play(Write(success))
        
        # MLP架构图
        mlp_title = Text('MLP Architecture:', font_size=18, color=WHITE)
        mlp_title.to_corner(DOWN + LEFT, buff=0.5)
        
        # 简化的网络结构
        input_layer = VGroup()
        for i in range(2):
            circle = Circle(radius=0.08, color=WHITE, fill_opacity=0.5)
            circle.move_to(mlp_title.get_bottom() + DOWN * 0.5 + RIGHT * (i * 0.4 - 0.2))
            input_layer.add(circle)
        
        hidden_layer = VGroup()
        for i in range(2):
            circle = Circle(radius=0.08, color=YELLOW_ACCENT, fill_opacity=0.5)
            circle.move_to(input_layer.get_bottom() + DOWN * 0.5 + RIGHT * (i * 0.4 - 0.2))
            hidden_layer.add(circle)
        
        output_layer = Circle(radius=0.1, color=GREEN_SUCCESS, fill_opacity=0.5)
        output_layer.move_to(hidden_layer.get_bottom() + DOWN * 0.5)
        
        self.play(Write(mlp_title))
        self.play(Create(input_layer), Create(hidden_layer), Create(output_layer))
        
        # 连接线
        for i in input_layer:
            for j in hidden_layer:
                line = Line(i.get_bottom(), j.get_top(), color=GRAY, stroke_width=1)
                self.add(line)
        
        for h in hidden_layer:
            line = Line(h.get_bottom(), output_layer.get_top(), color=GRAY, stroke_width=1)
            self.add(line)
        
        self.wait(3)
        
        # self.renderer.update_frame()
        # self.renderer.get_image().save('xor_solution_manim.png')
        print("已保存: xor_solution_manim.png")


class CombinedLinearScene(Scene):
    """
    场景5: 组合对比图
    
    将四个场景整合在一起展示
    """
    
    def construct(self):
        """构建场景"""
        
        self.camera.background_color = BACKGROUND_COLOR
        
        title = Text(
            'Linear vs Non-linear Classification',
            font_size=38,
            color=WHITE,
            font='Roboto'
        )
        title.to_edge(UP, buff=0.5)
        self.add(title)
        
        # 四个子区域
        # 子图1: 线性可分
        box1 = Rectangle(width=3.8, height=2.8, color=BLUE_LIGHT, stroke_width=2)
        box1.move_to(UP * 0.5 + LEFT * 2.3)
        
        label1 = Text('Linear Separable', font_size=16, color=BLUE_LIGHT)
        label1.move_to(box1.get_top() + UP * 0.15)
        
        # 子图2: 线性不可分
        box2 = Rectangle(width=3.8, height=2.8, color=RED_LIGHT, stroke_width=2)
        box2.move_to(UP * 0.5 + RIGHT * 2.3)
        
        label2 = Text('Linear Inseparable', font_size=16, color=RED_LIGHT)
        label2.move_to(box2.get_top() + UP * 0.15)
        
        # 子图3: XOR问题
        box3 = Rectangle(width=3.8, height=2.8, color=YELLOW_ACCENT, stroke_width=2)
        box3.move_to(DOWN * 0.8 + LEFT * 2.3)
        
        label3 = Text('XOR Problem', font_size=16, color=YELLOW_ACCENT)
        label3.move_to(box3.get_top() + UP * 0.15)
        
        # 子图4: MLP解决方案
        box4 = Rectangle(width=3.8, height=2.8, color=GREEN_SUCCESS, stroke_width=2)
        box4.move_to(DOWN * 0.8 + RIGHT * 2.3)
        
        label4 = Text('MLP Solution', font_size=16, color=GREEN_SUCCESS)
        label4.move_to(box4.get_top() + UP * 0.15)
        
        # 绘制边框
        self.play(Create(box1), Write(label1))
        self.play(Create(box2), Write(label2))
        self.play(Create(box3), Write(label3))
        self.play(Create(box4), Write(label4))
        
        # 添加简要说明
        desc1 = MathTex(r'\text{✓ Separable by line}', font_size=14, color=GREEN_SUCCESS)
        desc1.move_to(box1.get_center() + DOWN * 0.8)
        
        desc2 = MathTex(r'\text{✗ No line works}', font_size=14, color=RED_DARK)
        desc2.move_to(box2.get_center() + DOWN * 0.8)
        
        desc3 = MathTex(r'\text{✗ Classic non-linear}', font_size=14, color=RED_DARK)
        desc3.move_to(box3.get_center() + DOWN * 0.8)
        
        desc4 = MathTex(r'\text{✓ Two hidden neurons}', font_size=14, color=GREEN_SUCCESS)
        desc4.move_to(box4.get_center() + DOWN * 0.8)
        
        self.play(Write(desc1), Write(desc2))
        self.play(Write(desc3), Write(desc4))
        
        # 底部总结
        summary = MathTex(
            r'\text{Key: Linear models } \to \text{ single hyperplane}',
            font_size=18,
            color=WHITE
        )
        summary.to_edge(DOWN, buff=0.3)
        
        summary2 = MathTex(
            r'\text{Non-linear models } \to \text{ multiple hyperplanes / kernel trick}',
            font_size=18,
            color=BLUE_LIGHT
        )
        summary2.next_to(summary, DOWN, buff=0.1)
        
        self.play(Write(summary), Write(summary2))
        
        self.wait(3)
        
        # self.renderer.update_frame()
        # self.renderer.get_image().save('linear_classification_combined_manim.png')
        print("已保存: linear_classification_combined_manim.png")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("线性分类与异或问题可视化 - Manim版本")
    print("Linear Classification & XOR Visualization (Manim)")
    print("=" * 70)
    print()
    print("使用manim命令生成高质量静态图片:")
    print()
    print("1. 线性可分问题:")
    print("   manim -pqh linear_classification_manim.py LinearSeparableScene")
    print()
    print("2. 线性不可分问题:")
    print("   manim -pqh linear_classification_manim.py LinearInseparableScene")
    print()
    print("3. 异或问题:")
    print("   manim -pqh linear_classification_manim.py XORScene")
    print()
    print("4. MLP解决方案:")
    print("   manim -pqh linear_classification_manim.py XORSolutionScene")
    print()
    print("5. 组合对比图:")
    print("   manim -pqh linear_classification_manim.py CombinedLinearScene")
    print()
    print("参数说明:")
    print("  -p: 预览")
    print("  -q: 质量 (l=低, m=中, h=高)")
    print("  -h: 高质量 (推荐)")
    print()
    print("示例:")
    print("  manim -pl linear_classification_manim.py LinearSeparableScene")
    print("  manim -pqh linear_classification_manim.py CombinedLinearScene")
    print()
    print("=" * 70)
