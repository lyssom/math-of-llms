"""
激活函数概率映射可视化 - Sigmoid与Softmax的概率解释
Activation Functions Probability Mapping: Sigmoid & Softmax Visualization

使用方法：
1. 安装 Manim: pip install manim
2. 运行渲染: manim -pql activation_probability_mapping.py ProbabilityMappingScene
   - -p: 预览
   - -ql: 低质量（快速测试）
   - -qh: 高质量（最终输出）

作者：MiniMax Agent
"""

from manim import *

class ProbabilityMappingScene(Scene):
    """概率映射可视化主场景"""
    
    def construct(self):
        # ==================== 配置部分 ====================
        self.setup_colors()
        
        # ==================== 场景1：标题与介绍 ====================
        self.show_title()
        
        # ==================== 场景2：Sigmoid映射 ====================
        self.visualize_sigmoid_mapping()
        
        # ==================== 场景3：Softmax映射 ====================
        self.visualize_softmax_mapping()
        
        # ==================== 场景4：概率单纯形 ====================
        self.visualize_probability_simplex()
        
        # ==================== 场景5：总结 ====================
        self.show_summary()
        
        # 结束
        self.wait(2)
    
    def setup_colors(self):
        """设置颜色方案"""
        self.COLOR_SIGMOID = BLUE          # Sigmoid - 蓝色
        self.COLOR_SOFTMAX = GREEN         # Softmax - 绿色
        self.COLOR_LOGITS = ORANGE         # Logits - 橙色
        self.COLOR_PROBABILITY = PURPLE    # 概率 - 紫色
        self.COLOR_AXIS = GREY             # 坐标轴 - 灰色
        self.COLOR_TEXT = WHITE            # 文本 - 白色
        self.BG_COLOR = "#1a1a2e"          # 背景色 - 深蓝黑
        
        # 激活函数颜色
        self.COLOR_RELU = RED
        self.COLOR_GELU = TEAL
        self.COLOR_TANH = PINK
    
    def show_title(self):
        """场景1：显示标题"""
        self.camera.background_color = self.BG_COLOR
        
        # 主标题
        title = Text("激活函数的概率映射", font_size=36, color=self.COLOR_TEXT)
        title.to_edge(UP, buff=0.8)
        
        # 副标题
        subtitle = Text("Sigmoid与Softmax的概率解释", font_size=24, color=self.COLOR_PROBABILITY)
        subtitle.next_to(title, DOWN, buff=0.3)
        
        self.play(Write(title), run_time=1)
        self.play(Write(subtitle), run_time=1)
        
        # 简单介绍
        intro = Text("从Logits到概率的数学变换", font_size=18, color=GREY)
        intro.next_to(subtitle, DOWN, buff=0.5)
        
        self.play(Write(intro), run_time=1)
        self.wait(1)
        
        # 淡出
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(intro),
            run_time=0.5
        )
    
    def visualize_sigmoid_mapping(self):
        """场景2：可视化Sigmoid映射"""
        # 标题
        title = Text("Sigmoid函数：Logits → 二分类概率", font_size=28, color=self.COLOR_SIGMOID)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title), run_time=1)
        
        # 创建坐标轴
        axes = Axes(
            x_range=[-8, 8, 2],
            y_range=[0, 1.2, 0.5],
            axis_config={
                "color": self.COLOR_AXIS,
                "include_tip": True,
            },
            x_axis_config={
                "length": 8,
                "label_direction": DOWN,
            },
            y_axis_config={
                "length": 5,
                "label_direction": LEFT,
            }
        )
        
        axes.move_to(DOWN * 0.5)
        
        # 绘制Sigmoid曲线
        sigmoid_curve = axes.plot(
            lambda x: 1 / (1 + np.exp(-x)),
            x_range=[-8, 8],
            color=self.COLOR_SIGMOID,
            stroke_width=4
        )
        
        # 添加坐标轴标签
        x_label = axes.get_x_axis_label("Logits (z)", direction=DOWN, buff=0.3)
        y_label = axes.get_y_axis_label("Probability p", direction=LEFT, buff=0.3)
        
        x_label.set_color(self.COLOR_LOGITS)
        y_label.set_color(self.COLOR_PROBABILITY)
        
        self.play(
            Write(axes),
            Write(x_label),
            Write(y_label),
            run_time=1.5
        )
        
        # 绘制曲线
        self.play(
            Write(sigmoid_curve),
            run_time=2
        )
        
        # 添加关键点标注
        # z = 0, p = 0.5
        point_05 = Dot(
            axes.coords_to_point(0, 0.5),
            color=self.COLOR_PROBABILITY,
            radius=0.1
        )
        label_05 = MathTex(r"p=0.5", font_size=16, color=self.COLOR_PROBABILITY)
        label_05.next_to(point_05, UP, buff=0.1)
        
        # z → ∞, p → 1
        point_1 = Dot(
            axes.coords_to_point(6, 1),
            color=self.COLOR_PROBABILITY,
            radius=0.1
        )
        label_1 = MathTex(r"p \to 1", font_size=16, color=self.COLOR_PROBABILITY)
        label_1.next_to(point_1, UP + RIGHT, buff=0.1)
        
        # z → -∞, p → 0
        point_0 = Dot(
            axes.coords_to_point(-6, 0),
            color=self.COLOR_PROBABILITY,
            radius=0.1
        )
        label_0 = MathTex(r"p \to 0", font_size=16, color=self.COLOR_PROBABILITY)
        label_0.next_to(point_0, DOWN + LEFT, buff=0.1)
        
        self.play(
            Write(point_05),
            Write(label_05),
            Write(point_1),
            Write(label_1),
            Write(point_0),
            Write(label_0),
            run_time=1.5
        )
        
        # Sigmoid公式
        sigmoid_formula = MathTex(
            r"\sigma(z) = \frac{1}{1 + e^{-z}}",
            font_size=32,
            color=self.COLOR_SIGMOID
        )
        sigmoid_formula.to_corner(DR, buff=0.5)
        
        self.play(Write(sigmoid_formula), run_time=1)
        
        self.wait(1.5)
        
        # 保存引用
        self.sigmoid_elements = VGroup(
            axes, sigmoid_curve, x_label, y_label,
            point_05, label_05, point_1, label_1, point_0, label_0,
            sigmoid_formula
        )
    
    def visualize_softmax_mapping(self):
        """场景3：可视化Softmax映射"""
        # 清理前一个场景
        self.play(
            FadeOut(self.sigmoid_elements),
            run_time=0.5
        )
        
        # 新标题
        title = Text("Softmax函数：多维Logits → 概率分布", font_size=28, color=self.COLOR_SOFTMAX)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title), run_time=1)
        
        # 创建Logits向量可视化
        logits_group = VGroup()
        
        # Logits向量标签
        logits_label = Text("Logits向量 z", font_size=20, color=self.COLOR_LOGITS)
        logits_label.to_edge(LEFT, buff=1).shift(UP * 1.5)
        
        # Logits矩形条形图
        num_classes = 4
        bar_width = 1.0
        bar_spacing = 0.5
        max_logit = 3.0
        
        logits_bars = VGroup()
        logits_values = [2.5, 1.0, -0.5, 0.5]  # 示例logits值
        
        for i, value in enumerate(logits_values):
            # 绘制条形
            bar = Rectangle(
                width=bar_width,
                height=value * 0.8,  # 缩放因子
                fill_opacity=0.8,
                color=self.COLOR_LOGITS
            )
            
            # 正值绿色，负值红色
            if value >= 0:
                bar.set_fill(self.COLOR_SOFTMAX)
            else:
                bar.set_fill(RED)
            
            bar.move_to(ORIGIN + RIGHT * (i * (bar_width + bar_spacing)))
            bar.shift(DOWN * (1 - value * 0.4))  # 底部对齐
            
            # 条形标签
            label = MathTex(f"z_{i+1}", font_size=14, color=self.COLOR_LOGITS)
            label.next_to(bar, DOWN, buff=0.1)
            
            logits_bars.add(bar, label)
        
        logits_group.add(logits_label, logits_bars)
        logits_group.move_to(UP * 1.5)
        
        self.play(Write(logits_group), run_time=1.5)
        
        # 箭头
        arrow = Arrow(
            start=logits_group.get_bottom() + DOWN * 0.3,
            end=logits_group.get_bottom() + DOWN * 1.5,
            color=WHITE,
            stroke_width=2
        )
        
        self.play(Write(arrow), run_time=0.5)
        
        # Softmax公式
        softmax_formula = MathTex(
            r"\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}",
            font_size=28,
            color=self.COLOR_SOFTMAX
        )
        softmax_formula.move_to(DOWN)
        
        self.play(Write(softmax_formula), run_time=1)
        
        self.wait(1.5)
        
        # 保存引用
        self.softmax_elements = VGroup(title, logits_group, arrow, softmax_formula)
    
    def visualize_probability_simplex(self):
        """场景4：可视化概率单纯形"""
        # 清理
        self.play(
            FadeOut(self.softmax_elements),
            run_time=0.5
        )
        
        # 标题
        title = Text("概率单纯形：Softmax输出的几何表示", font_size=24, color=self.COLOR_PROBABILITY)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title), run_time=1)
        
        # 创建概率单纯形（三角形表示）
        # 使用二维概率分布的三角形图
        simplex_size = 3.0
        
        # 三角形顶点
        top = UP * 2.5
        bottom_left = LEFT * 2 + DOWN * 1.5
        bottom_right = RIGHT * 2 + DOWN * 1.5
        
        # 绘制三角形
        triangle = Polygon(
            top, bottom_left, bottom_right,
            color=self.COLOR_PROBABILITY,
            stroke_width=3,
            fill_opacity=0.1
        )
        
        # 顶点标签
        label_p1 = MathTex(r"p_1", font_size=20, color=self.COLOR_SOFTMAX)
        label_p1.next_to(top, UP, buff=0.2)
        
        label_p2 = MathTex(r"p_2", font_size=20, color=self.COLOR_SOFTMAX)
        label_p2.next_to(bottom_left, DOWN + LEFT, buff=0.2)
        
        label_p3 = MathTex(r"p_3", font_size=20, color=self.COLOR_SOFTMAX)
        label_p3.next_to(bottom_right, DOWN + RIGHT, buff=0.2)
        
        # 内部点表示示例分布
        # 例如：p = [0.6, 0.3, 0.1]
        point_p = Dot(
            top * 0.6 + bottom_left * 0.3 + bottom_right * 0.1,
            color=self.COLOR_SIGMOID,
            radius=0.15
        )
        
        label_point = MathTex(r"\hat{p} = [0.6, 0.3, 0.1]", font_size=16, color=self.COLOR_SIGMOID)
        label_point.next_to(point_p, RIGHT, buff=0.3)
        
        # 概率约束说明
        constraints = VGroup()
        constraint1 = Text("p_i > 0 (正定性)", font_size=14, color=GREEN_A)
        constraint2 = Text("∑p_i = 1 (归一化)", font_size=14, color=GREEN_A)
        
        constraints.add(constraint1, constraint2)
        constraints.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        constraints.to_corner(DR, buff=0.5)
        
        # 绘制单纯形
        self.play(
            Write(triangle),
            Write(label_p1),
            Write(label_p2),
            Write(label_p3),
            run_time=1.5
        )
        
        # 绘制示例点
        self.play(
            Write(point_p),
            Write(label_point),
            run_time=1
        )
        
        # 绘制约束条件
        self.play(Write(constraints), run_time=1)
        
        # 内部网格线
        grid_lines = VGroup()
        for i in range(1, 10):
            ratio = i / 10
            # 平行于底边的线
            line_start = top * ratio + bottom_left * (1 - ratio)
            line_end = top * ratio + bottom_right * (1 - ratio)
            grid_line = Line(line_start, line_end, color=GREY, stroke_width=1, stroke_opacity=0.3)
            grid_lines.add(grid_line)
        
        self.play(Write(grid_lines), run_time=1)
        
        # 从logits到单纯形的映射说明
        mapping_text = Text("指数化 → 归一化", font_size=18, color=self.COLOR_LOGITS)
        mapping_text.next_to(triangle, LEFT, buff=0.5)
        
        self.play(Write(mapping_text), run_time=0.8)
        
        self.wait(2)
        
        # 保存引用
        self.simplex_elements = VGroup(
            triangle, label_p1, label_p2, label_p3,
            point_p, label_point, constraints, grid_lines, mapping_text
        )
    
    def show_summary(self):
        """场景5：总结"""
        # 清理
        self.play(
            FadeOut(self.simplex_elements),
            run_time=0.5
        )
        
        # 总结标题
        summary_title = Text("概率映射总结", font_size=32, color=self.COLOR_PROBABILITY)
        summary_title.to_edge(UP, buff=0.5)
        
        self.play(Write(summary_title), run_time=1)
        
        # 总结要点
        summary_points = VGroup()
        
        point1 = Text("• Sigmoid: R → (0, 1) - 二分类概率", font_size=20, color=self.COLOR_SIGMOID)
        point2 = Text("• Softmax: R^C → Δ^(C-1) - 多分类分布", font_size=20, color=self.COLOR_SOFTMAX)
        point3 = Text("• Logits是无界实数，概率是归一化值", font_size=20, color=self.COLOR_LOGITS)
        point4 = Text("• Softmax输出满足概率分布公理", font_size=20, color=self.COLOR_PROBABILITY)
        
        summary_points.add(point1, point2, point3, point4)
        summary_points.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        summary_points.center()
        
        self.play(Write(summary_points), run_time=2)
        
        # 最终动画
        self.play(
            summary_points.animate.scale(1.1),
            run_time=0.5,
            rate_func=there_and_back
        )
        
        self.wait(2)
        
        # 淡出
        self.play(
            FadeOut(summary_title),
            FadeOut(summary_points),
            run_time=1
        )


# ==================== 额外场景：常见激活函数对比 ====================

class ActivationFunctionsComparison(Scene):
    """常见激活函数对比可视化"""
    
    def construct(self):
        self.setup_colors()
        self.camera.background_color = self.BG_COLOR
        
        # 标题
        title = Text("常见激活函数对比", font_size=32, color=self.COLOR_TEXT)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title), run_time=1)
        
        # 创建多坐标轴对比
        # Sigmoid
        sigmoid_axes = Axes(
            x_range=[-6, 6, 3],
            y_range=[-0.5, 1.5, 1],
            axis_config={"color": self.COLOR_AXIS, "include_tip": True},
            x_axis_config={"length": 4, "label_direction": DOWN},
            y_axis_config={"length": 3, "label_direction": LEFT}
        )
        sigmoid_axes.move_to(UP * 1.5 + LEFT * 3.5)
        
        sigmoid_curve = sigmoid_axes.plot(
            lambda x: 1 / (1 + np.exp(-x)),
            x_range=[-6, 6],
            color=self.COLOR_SIGMOID,
            stroke_width=3
        )
        
        sigmoid_label = Text("Sigmoid", font_size=16, color=self.COLOR_SIGMOID)
        sigmoid_label.next_to(sigmoid_axes, UP, buff=0.2)
        
        # ReLU
        relu_axes = Axes(
            x_range=[-3, 3, 2],
            y_range=[-1, 4, 1],
            axis_config={"color": self.COLOR_AXIS, "include_tip": True},
            x_axis_config={"length": 4, "label_direction": DOWN},
            y_axis_config={"length": 3, "label_direction": LEFT}
        )
        relu_axes.move_to(UP * 1.5 + RIGHT * 3.5)
        
        relu_curve = relu_axes.plot(
            lambda x: max(0, x),
            x_range=[-3, 3],
            color=self.COLOR_RELU,
            stroke_width=3
        )
        
        relu_label = Text("ReLU", font_size=16, color=self.COLOR_RELU)
        relu_label.next_to(relu_axes, UP, buff=0.2)
        
        # Tanh
        tanh_axes = Axes(
            x_range=[-6, 6, 3],
            y_range=[-2, 2, 1],
            axis_config={"color": self.COLOR_AXIS, "include_tip": True},
            x_axis_config={"length": 4, "label_direction": DOWN},
            y_axis_config={"length": 3, "label_direction": LEFT}
        )
        tanh_axes.move_to(DOWN * 1.5 + LEFT * 3.5)
        
        tanh_curve = tanh_axes.plot(
            lambda x: np.tanh(x),
            x_range=[-6, 6],
            color=self.COLOR_TANH,
            stroke_width=3
        )
        
        tanh_label = Text("Tanh", font_size=16, color=self.COLOR_TANH)
        tanh_label.next_to(tanh_axes, UP, buff=0.2)
        
        # GELU
        gelu_axes = Axes(
            x_range=[-4, 4, 2],
            y_range=[-1, 4, 1],
            axis_config={"color": self.COLOR_AXIS, "include_tip": True},
            x_axis_config={"length": 4, "label_direction": DOWN},
            y_axis_config={"length": 3, "label_direction": LEFT}
        )
        gelu_axes.move_to(DOWN * 1.5 + RIGHT * 3.5)
        
        # GELU近似公式
        def gelu(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        
        gelu_curve = gelu_axes.plot(
            gelu,
            x_range=[-4, 4],
            color=self.COLOR_GELU,
            stroke_width=3
        )
        
        gelu_label = Text("GELU", font_size=16, color=self.COLOR_GELU)
        gelu_label.next_to(gelu_axes, UP, buff=0.2)
        
        # 绘制所有图表
        self.play(
            Write(sigmoid_axes),
            Write(sigmoid_curve),
            Write(sigmoid_label),
            Write(relu_axes),
            Write(relu_curve),
            Write(relu_label),
            Write(tanh_axes),
            Write(tanh_curve),
            Write(tanh_label),
            Write(gelu_axes),
            Write(gelu_curve),
            Write(gelu_label),
            run_time=2
        )
        
        self.wait(2)


# ==================== 场景运行配置 ====================

if __name__ == "__main__":
    print("请使用以下命令渲染动画：")
    print("主场景 - 概率映射:")
    print("  manim -pql activation_probability_mapping.py ProbabilityMappingScene")
    print("\n激活函数对比场景:")
    print("  manim -pql activation_probability_mapping.py ActivationFunctionsComparison")
    print("\n质量选项：")
    print("  -ql: 低质量（快速测试）")
    print("  -qm: 中等质量")
    print("  -qh: 高质量（最终输出）")
