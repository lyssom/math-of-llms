"""
Manim visualization of Information Geometry on Probability Simplex
展示概率单纯形上的信息几何结构 - 清晰展示KL散度和Fisher信息度量

Key concepts visualized:
1. 2D概率单纯形（三角形表示）
2. KL散度等高线（从真实分布q发散）
3. Fisher信息度量（紫色椭球表示局部黎曼几何）
4. 投影过程（从初始模型到真实分布的优化轨迹）
"""

from manim import *
import numpy as np

class InfoGeometryVisualization(Scene):
    def construct(self):
        # 设置深色背景，突出显示图形
        self.camera.background_color = "#0d1117"
        
        # ==========================================
        # 第一部分：标题和说明
        # ==========================================
        
        title = Text(
            "Information Geometry on Probability Simplex",
            font_size=28,
            color=WHITE,
            weight=BOLD
        ).to_edge(UP, buff=0.5)
        
        subtitle = Text(
            "KL Divergence & Fisher Information Metric",
            font_size=18,
            color=GRAY
        ).next_to(title, DOWN, buff=0.2)
        
        # ==========================================
        # 第二部分：绘制概率单纯形（2D表示）
        # ==========================================
        
        # 单纯形顶点坐标
        v1 = np.array([-3, -1.5, 0])  # P = (1,0,0)
        v2 = np.array([3, -1.5, 0])   # P = (0,1,0)
        v3 = np.array([0, 2.5, 0])    # P = (0,0,1)
        
        # 绘制单纯形边界（蓝色线条）
        simplex = Polygon(v1, v2, v3, color=BLUE_C, fill_opacity=0.1, stroke_width=4)
        
        # 顶点标签
        label1 = Text("P1=(1,0,0)", font_size=16, color=RED).next_to(v1, DOWN, buff=0.3)
        label2 = Text("P2=(0,1,0)", font_size=16, color=GREEN).next_to(v2, DOWN, buff=0.3)
        label3 = Text("P3=(0,0,1)", font_size=16, color=BLUE).next_to(v3, UP, buff=0.3)
        
        # 单纯形内部标签
        interior_label = Text("Valid Probability Distributions\n(p1+p2+p3=1)", 
                            font_size=14, color=BLUE_C, opacity=0.8)
        interior_label.move_to(np.array([0, 0, 0]))
        
        # ==========================================
        # 第三部分：KL散度等高线
        # ==========================================
        
        # 真实分布 q = [0.4, 0.35, 0.25]
        # 转换为2D坐标
        def prob_to_2d(prob):
            """将3D概率向量转换为单纯形内的2D坐标"""
            return v1 + prob[1] * (v2 - v1) + prob[2] * (v3 - v1)
        
        q = np.array([0.4, 0.35, 0.25])
        q_pos = prob_to_2d(q)
        
        # 绘制KL散度等高线（同心椭圆近似）
        kl_contours = VGroup()
        levels = [0.05, 0.15, 0.3, 0.5, 0.8]
        colors = [GREEN, YELLOW, ORANGE, RED, PURPLE]
        
        for level, color in zip(levels, colors):
            # 使用椭圆近似等高线
            ellipse = Ellipse(
                width=0.8 + level * 4,
                height=0.5 + level * 2.5,
                color=color,
                stroke_width=2,
                fill_opacity=0.05 + level * 0.1
            ).move_to(q_pos)
            kl_contours.add(ellipse)
        
        # 真实分布点
        true_dist = Dot(q_pos, radius=0.15, color=YELLOW)
        # 添加发光效果（使用更大的半透明圆）
        true_glow = Circle(radius=0.25, color=YELLOW, fill_opacity=0.3).move_to(q_pos)
        true_label = Text("True Distribution q", font_size=16, color=YELLOW, weight=BOLD)
        true_label.next_to(true_dist, RIGHT, buff=0.3)
        
        # KL散度公式（使用文本代替LaTeX）
        kl_formula = Text(
            "DKL(P || q) = sum(Pi * log(Pi/qi))",
            font_size=18,
            color=ORANGE
        ).to_edge(DOWN, buff=1.5)
        
        # ==========================================
        # 第四部分：Fisher信息度量（黎曼几何）
        # ==========================================
        
        # 在单纯形内部选择几个代表性点绘制Fisher信息椭球
        fisher_points = [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.3, 0.4, 0.3]),
            np.array([0.6, 0.25, 0.15]),
            np.array([0.35, 0.3, 0.35]),
        ]
        
        fisher_ellipsoids = VGroup()
        for i, p in enumerate(fisher_points):
            pos = prob_to_2d(p)
            # 不同方向的椭球表示Fisher信息矩阵的各向异性
            ellipse = Ellipse(
                width=0.5 + i * 0.15,
                height=0.35 + i * 0.1,
                color=PURPLE,
                stroke_width=2,
                fill_opacity=0.2
            ).rotate(i * 0.3).move_to(pos)
            fisher_ellipsoids.add(ellipse)
        
        fisher_label = Text("Fisher Information Ellipsoids\n(Riemannian Metric)", 
                          font_size=14, color=PURPLE, weight=BOLD)
        fisher_label.next_to(fisher_ellipsoids, LEFT, buff=0.3)
        
        # ==========================================
        # 第五部分：投影和优化过程
        # ==========================================
        
        # 初始模型分布
        p_init = np.array([0.7, 0.2, 0.1])
        p_init_pos = prob_to_2d(p_init)
        
        # 优化轨迹点
        trajectory_points = []
        for t in np.linspace(0, 1, 20):
            # 在KL散度度量下的近似测地线路径
            straight = (1-t) * p_init_pos + t * q_pos
            # 添加曲率效果（KL几何不是欧几里得的）
            mid_point = (1-t) * p_init + t * q
            curvature = 0.15 * np.sin(np.pi * t) * 0.3
            # 计算垂直于单纯形平面的方向偏移
            offset = np.array([-curvature * 0.5, curvature, 0])
            trajectory_points.append(straight + offset)
        
        # 绘制优化轨迹
        trajectory = VMobject(color=ORANGE, stroke_width=4)
        trajectory.set_points_smoothly(trajectory_points)
        
        # 起点
        start_point = Dot(p_init_pos, radius=0.12, color=ORANGE)
        # 添加发光效果
        start_glow = Circle(radius=0.2, color=ORANGE, fill_opacity=0.3).move_to(p_init_pos)
        start_label = Text("Initial Model P0", font_size=14, color=ORANGE, weight=BOLD)
        start_label.next_to(start_point, DOWN, buff=0.2)
        
        # 终点（收敛到真实分布）
        end_point = Dot(q_pos, radius=0.12, color=GREEN)
        # 添加发光效果
        end_glow = Circle(radius=0.2, color=GREEN, fill_opacity=0.3).move_to(q_pos)
        end_label = Text("Converged to q", font_size=14, color=GREEN, weight=BOLD)
        end_label.next_to(end_point, UP, buff=0.2)
        
        # 投影箭头
        projection_arrow = Arrow(
            start=start_point.get_center() + np.array([0, 0.2, 0]),
            end=end_point.get_center() - np.array([0, 0.2, 0]),
            color=ORANGE,
            stroke_width=3,
            buff=0.2
        )
        projection_text = Text("KL Projection\n(Information Geometry)", 
                             font_size=12, color=ORANGE, weight=BOLD)
        projection_text.move_to((start_point.get_center() + end_point.get_center()) / 2 + np.array([0.8, 0, 0]))
        
        # ==========================================
        # 第六部分：欧氏 vs 黎曼几何对比
        # ==========================================
        
        # 欧氏直线（虚线）
        euclidean_line = DashedLine(
            p_init_pos + np.array([0, 0.3, 0]),
            q_pos - np.array([0, 0.3, 0]),
            color=GRAY,
            stroke_width=2,
            dash_length=0.15
        )
        euclidean_label = Text("Euclidean Geodesic\n(Not Optimal)", 
                              font_size=11, color=GRAY)
        euclidean_label.next_to(euclidean_line, LEFT, buff=0.2)
        
        # 黎曼测地线标签
        riemann_label = Text("Riemannian Geodesic\n(KL-induced Metric)", 
                           font_size=11, color=GREEN, weight=BOLD)
        riemann_label.next_to(trajectory, RIGHT, buff=0.5)
        
        # ==========================================
        # 第七部分：Fisher信息公式
        # ==========================================
        
        # Fisher信息公式（使用文本代替LaTeX）
        fisher_formula = Text(
            "g_ij(P) = E[dlogP/dthetai * dlogP/dthetaj]",
            font_size=16,
            color=PURPLE
        )
        fisher_formula.move_to(np.array([4.5, -2.5, 0]))
        
        formula_bg = Rectangle(
            width=4,
            height=1,
            color=PURPLE,
            stroke_width=1,
            fill_opacity=0.1
        ).move_to(fisher_formula.get_center())
        
        # ==========================================
        # 组合所有元素
        # ==========================================
        
        # 分层组合
        layer1 = VGroup(title, subtitle)  # 标题层
        layer2 = VGroup(simplex, label1, label2, label3, interior_label)  # 单纯形层
        layer3 = VGroup(kl_contours, true_dist, true_glow, true_label)  # KL散度层
        layer4 = VGroup(fisher_ellipsoids, fisher_label)  # Fisher信息层
        layer5 = VGroup(start_point, start_glow, start_label, end_point, end_glow, end_label, 
                       trajectory, projection_arrow, projection_text)  # 投影层
        layer6 = VGroup(euclidean_line, euclidean_label, riemann_label)  # 几何对比层
        layer7 = VGroup(formula_bg, fisher_formula, kl_formula)  # 公式层
        
        # 添加到场景
        self.add(layer1)
        self.add(layer2)
        self.add(layer3)
        self.add(layer4)
        self.add(layer5)
        self.add(layer6)
        self.add(layer7)


# 运行命令: manim -pql info_geometry_visualization.py InfoGeometryVisualization
