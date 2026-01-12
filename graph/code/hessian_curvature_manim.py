"""
Hessian矩阵与曲率关系可视化
Hessian Matrix and Curvature Relationship Visualization

用于《大模型中的数学》第一章第3节微积分与优化基础
展示三种典型的Hessian矩阵特征：
1. 正定 Hessian → 局部极小值点（碗状曲面）
2. 负定 Hessian → 局部极大值点（倒碗状曲面）
3. 混合特征值 → 鞍点（一个方向凸，一个方向凹）

作者: MiniMax Agent

使用说明:
1. 安装依赖: pip install manim
2. 生成静态图片: python hessian_curvature_manim.py
3. 或运行单个场景:
   manim -pqh hessian_curvature_manim.py PositiveHessianScene
   manim -pqh hessian_curvature_manim.py NegativeHessianScene
   manim -pqh hessian_curvature_manim.py SaddlePointScene
   manim -pqh hessian_curvature_manim.py CombinedScene
"""

import numpy as np
from manim import *



my_template = TexTemplate(
    tex_compiler="xelatex",
    output_format=".xdv"
)
my_template.add_to_preamble(r"\usepackage{xeCJK}")
my_template.add_to_preamble(r"\setCJKmainfont{SimSun}")
# ============================================================================
# 配置参数
# ============================================================================

# 图像分辨率
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30

# ============================================================================
# 数学函数定义
# ============================================================================

def convex_surface(x, y):
    """
    凸曲面（正定Hessian）: f(x,y) = x² + y²
    
    Hessian矩阵:
    H = [[2, 0],
         [0, 2]]
    
    特征值: λ₁ = 2, λ₂ = 2 (均为正)
    → 局部极小值点
    """
    return x**2 + y**2

def concave_surface(x, y):
    """
    凹曲面（负定Hessian）: f(x,y) = -(x² + y²)
    
    Hessian矩阵:
    H = [[-2, 0],
         [0, -2]]
    
    特征值: λ₁ = -2, λ₂ = -2 (均为负)
    → 局部极大值点
    """
    return -(x**2 + y**2)

def saddle_surface(x, y):
    """
    鞍点曲面（混合Hessian）: f(x,y) = x² - y²
    
    Hessian矩阵:
    H = [[2, 0],
         [0, -2]]
    
    特征值: λ₁ = 2 (正), λ₂ = -2 (负)
    → 鞍点：一个方向极小，一个方向极大
    """
    return x**2 - y**2

def get_hessian_eigenvalues(func_name):
    """
    获取指定函数的Hessian矩阵特征值
    """
    if func_name == "convex":
        return [2, 2]  # 均为正
    elif func_name == "concave":
        return [-2, -2]  # 均为负
    elif func_name == "saddle":
        return [2, -2]  # 一正一负
    return [0, 0]

# ============================================================================
# 场景类定义
# ============================================================================

class PositiveHessianScene(ThreeDScene):
    """
    场景1: 正定Hessian → 局部极小值点
    
    曲面形状: 碗状（向上凸起）
    Hessian特征值: λ₁ > 0, λ₂ > 0
    极值类型: 局部极小值点 ✓
    """
    
    def construct(self):
        """构建场景"""
        
        # 设置相机视角
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=45 * DEGREES,
            focal_distance=2
        )
        
        # 生成曲面数据
        x_range = np.linspace(-2, 2, 40)
        y_range = np.linspace(-2, 2, 40)
        X, Y = np.meshgrid(x_range, y_range)
        Z = convex_surface(X, Y)
        
        # 创建标题
        title = Text(
            'Positive Definite Hessian\n'
            r'$f(x,y) = x^2 + y^2$',
            font_size=36,
            color=BLUE
        )
        title.to_edge(UP, buff=0.5)
        self.add(title)
        
        # 创建3D曲面
        surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                convex_surface(u, v) / 2  # 缩放Z轴以适应显示
            ]),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(40, 40)
        )
        surface.set_style(
            fill_opacity=0.8,
            stroke_color=WHITE,
            stroke_width=0.5
        )
        surface.set_fill_by_value(
            axes=Z/2,
            cmap='viridis',
            max=2,
            min=0
        )
        
        self.play(Create(surface), run_time=2)
        
        # 添加曲率方向指示
        self.add_curvature_indicators("positive")
        
        # 添加Hessian信息
        hessian_info = MathTex(
            r'\mathbf{H} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}',
            font_size=28,
            color=YELLOW
        )
        hessian_info.to_corner(DOWN + LEFT, buff=0.5)
        
        eigenvalue_info = MathTex(
            r'\lambda_1 = 2,\ \lambda_2 = 2 \quad (\text{both positive})',
            font_size=24,
            color=GREEN
        )
        eigenvalue_info.next_to(hessian_info, DOWN, buff=0.3)
        
        conclusion = MathTex(
            r'\Rightarrow \text{local minimum}',
            font_size=28,
            color=RED
        )
        conclusion.next_to(eigenvalue_info, DOWN, buff=0.3)
        
        self.play(Write(hessian_info))
        self.play(Write(eigenvalue_info))
        self.play(Write(conclusion))
        
        # 标记原点（极小值点）
        min_point = Dot3D(
            point=[0, 0, 0],
            radius=0.1,
            color=RED_E
        )
        self.play(Create(min_point))
        
        self.wait(2)
        
        # 导出静态图片
        # self.renderer.update_frame()
        # self.renderer.get_image().save(
        #     'hessian_positive_definite.png'
        # )
        # frame = self.take_frame()
        # frame.save('hessian_positive_definite.png')
        # print("已保存: hessian_positive_definite.png")
        self.wait(2)
    
    def add_curvature_indicators(self, hessian_type):
        """添加曲率方向指示"""
        
        if hessian_type == "positive":
            # 向上弯曲的箭头表示正曲率
            arrow1 = Arrow3D(
                start=np.array([0, -1.5, 0.5]),
                end=np.array([0, -1.5, 1.5]),
                color=RED,
                stroke_width=3
            )
            arrow2 = Arrow3D(
                start=np.array([-1.5, 0, 0.5]),
                end=np.array([-1.5, 0, 1.5]),
                color=RED,
                stroke_width=3
            )
            
            label1 = Text("+曲率", font_size=16, color=RED)
            label1.move_to([0, -1.8, 1.8])
            label2 = Text("+曲率", font_size=16, color=RED)
            label2.move_to([-1.8, 0, 1.8])
            
            self.add(arrow1, arrow2, label1, label2)


class NegativeHessianScene(ThreeDScene):
    """
    场景2: 负定Hessian → 局部极大值点
    
    曲面形状: 倒碗状（向下凹陷）
    Hessian特征值: λ₁ < 0, λ₂ < 0
    极值类型: 局部极大值点
    """
    
    def construct(self):
        """构建场景"""
        
        # 设置相机视角
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=45 * DEGREES,
            focal_distance=2
        )
        
        # 生成曲面数据
        x_range = np.linspace(-2, 2, 40)
        y_range = np.linspace(-2, 2, 40)
        X, Y = np.meshgrid(x_range, y_range)
        Z = concave_surface(X, Y)
        
        # 创建标题
        title = Text(
            'Negative Definite Hessian\n'
            r'$f(x,y) = -(x^2 + y^2)$',
            font_size=36,
            color=BLUE
        )
        title.to_edge(UP, buff=0.5)
        self.add(title)
        
        # 创建3D曲面
        surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                concave_surface(u, v) / 2
            ]),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(40, 40)
        )
        surface.set_style(
            fill_opacity=0.8,
            stroke_color=WHITE,
            stroke_width=0.5
        )
        surface.set_fill_by_value(
            axes=Z/2,
            cmap='coolwarm',  # 使用冷暖色对比
            max=0,
            min=-2
        )
        
        self.play(Create(surface), run_time=2)
        
        # 添加Hessian信息
        hessian_info = MathTex(
            r'\mathbf{H} = \begin{bmatrix} -2 & 0 \\ 0 & -2 \end{bmatrix}',
            font_size=28,
            color=YELLOW
        )
        hessian_info.to_corner(DOWN + LEFT, buff=0.5)
        
        eigenvalue_info = MathTex(
            r'\lambda_1 = -2,\ \lambda_2 = -2 \quad (\text{均为负})',
            font_size=24,
            color=GREEN
        )
        eigenvalue_info.next_to(hessian_info, DOWN, buff=0.3)
        
        conclusion = MathTex(
            r'\Rightarrow \text{局部极大值点}',
            font_size=28,
            color=RED
        )
        conclusion.next_to(eigenvalue_info, DOWN, buff=0.3)
        
        self.play(Write(hessian_info))
        self.play(Write(eigenvalue_info))
        self.play(Write(conclusion))
        
        # 标记原点（极大值点）
        max_point = Dot3D(
            point=[0, 0, 0],
            radius=0.1,
            color=BLUE_E
        )
        self.play(Create(max_point))
        
        self.wait(2)
        
        # 导出静态图片
        # self.renderer.update_frame()
        # self.renderer.get_image().save(
        #     'hessian_negative_definite.png'
        # )
        print("已保存: hessian_negative_definite.png")


class SaddlePointScene(ThreeDScene):
    """
    场景3: 混合Hessian → 鞍点
    
    曲面形状: 马鞍面（一个方向凸，一个方向凹）
    Hessian特征值: λ₁ > 0, λ₂ < 0
    极值类型: 鞍点（非极值点）
    """
    
    def construct(self):
        """构建场景"""
        
        # 设置相机视角
        self.set_camera_orientation(
            phi=50 * DEGREES,
            theta=30 * DEGREES,
            focal_distance=2
        )
        
        # 生成曲面数据
        x_range = np.linspace(-2, 2, 40)
        y_range = np.linspace(-2, 2, 40)
        X, Y = np.meshgrid(x_range, y_range)
        Z = saddle_surface(X, Y)
        
        # 创建标题
        title = Text(
            'Saddle Point (Mixed Hessian)\n'
            r'$f(x,y) = x^2 - y^2$',
            font_size=36,
            color=BLUE
        )
        title.to_edge(UP, buff=0.5)
        self.add(title)
        
        # 创建3D曲面
        surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                saddle_surface(u, v) / 2
            ]),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(40, 40)
        )
        surface.set_style(
            fill_opacity=0.8,
            stroke_color=WHITE,
            stroke_width=0.5
        )
        surface.set_fill_by_value(
            axes=Z/2,
            cmap='RdYlBu_r',  # 红黄蓝渐变
            max=2,
            min=-2
        )
        
        self.play(Create(surface), run_time=2)
        
        # 添加方向指示
        # x方向（向上凸）
        arrow_x = Arrow3D(
            start=np.array([-1.5, 0, -0.3]),
            end=np.array([-1.5, 0, 0.7]),
            color=RED,
            stroke_width=3
        )
        label_x = Text("x方向: +曲率", font_size=14, color=RED)
        label_x.move_to([-1.5, 0.5, 1.0])
        
        # y方向（向下凹）
        arrow_y = Arrow3D(
            start=np.array([0, 1.5, 0.3]),
            end=np.array([0, 1.5, -0.7]),
            color=BLUE,
            stroke_width=3
        )
        label_y = Text("y方向: -曲率", font_size=14, color=BLUE)
        label_y.move_to([0.5, 1.5, -0.5])
        
        self.add(arrow_x, arrow_y, label_x, label_y)
        
        # 添加Hessian信息
        hessian_info = MathTex(
            r'\mathbf{H} = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}',
            font_size=28,
            color=YELLOW
        )
        hessian_info.to_corner(DOWN + LEFT, buff=0.5)
        
        eigenvalue_info = MathTex(
            r'\lambda_1 = 2\ (>0),\ \lambda_2 = -2\ (<0)',
            font_size=24,
            color=GREEN
        )
        eigenvalue_info.next_to(hessian_info, DOWN, buff=0.3)
        
        conclusion = MathTex(
            r'\Rightarrow \text{鞍点}',
            font_size=28,
            color=RED
        )
        conclusion.next_to(eigenvalue_info, DOWN, buff=0.3)
        
        self.play(Write(hessian_info))
        self.play(Write(eigenvalue_info))
        self.play(Write(conclusion))
        
        # 标记原点（鞍点）
        saddle_point = Dot3D(
            point=[0, 0, 0],
            radius=0.1,
            color=PURPLE
        )
        self.play(Create(saddle_point))
        
        self.wait(2)
        
        # 导出静态图片
        # self.renderer.update_frame()
        # self.renderer.get_image().save(
        #     'hessian_saddle_point.png'
        # )
        print("已保存: hessian_saddle_point.png")


class CombinedScene(ThreeDScene):
    """
    综合场景：三个曲面对比图
    
    将三种Hessian类型放在一个图中对比展示
    """
    
    def construct(self):
        """构建场景"""
        
        # 设置相机视角
        self.set_camera_orientation(
            phi=55 * DEGREES,
            theta=40 * DEGREES,
            focal_distance=3
        )
        
        # 标题
        title = Text(
            'Hessian Matrix and Curvature Relationship',
            font_size=32,
            color=BLACK
        )
        title.to_edge(UP, buff=0.3)
        self.add(title)
        
        # 创建三个曲面
        surfaces = []
        
        # 曲面1: 凸曲面（左侧）
        surface1 = Surface(
            lambda u, v: np.array([
                u - 3,  # 向左移动
                v,
                convex_surface(u, v) / 2.5
            ]),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=(25, 25)
        )
        surface1.set_style(
            fill_opacity=0.85,
            stroke_color=WHITE,
            stroke_width=0.3
        )
        surfaces.append(surface1)
        
        # 曲面2: 凹曲面（中间）
        surface2 = Surface(
            lambda u, v: np.array([
                u,  # 中间
                v,
                concave_surface(u, v) / 2.5
            ]),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=(25, 25)
        )
        surface2.set_style(
            fill_opacity=0.85,
            stroke_color=WHITE,
            stroke_width=0.3
        )
        surfaces.append(surface2)
        
        # 曲面3: 鞍面（右侧）
        surface3 = Surface(
            lambda u, v: np.array([
                u + 3,  # 向右移动
                v,
                saddle_surface(u, v) / 2.5
            ]),
            u_range=[-1.5, 1.5],
            v_range=[-1.5, 1.5],
            resolution=(25, 25)
        )
        surface3.set_style(
            fill_opacity=0.85,
            stroke_color=WHITE,
            stroke_width=0.3
        )
        # surfaces.append(surface3)
        
        # 依次创建曲面
        for surface in surfaces:
            self.play(Create(surface), run_time=1.5)
        
        # # 添加标签
        # label1 = MathTex(
        #     r'\text{Positive-definite Hessian}\\'
        #     r'\lambda_1 > 0,\ \lambda_2 > 0\\'
        #     r'\text{Local minimum}',
        #     font_size=18,
        #     color=GREEN,
        # )
        # label1.move_to(LEFT * 3 + DOWN * 1.8)

        # label2 = MathTex(
        #     r'\text{Negative-definite Hessian}\\'
        #     r'\lambda_1 < 0,\ \lambda_2 < 0\\'
        #     r'\text{Local maximum}',
        #     font_size=18,
        #     color=BLUE
        # )
        # label2.move_to(DOWN * 1.8)

        # label3 = MathTex(
        #     r'\text{Indefinite Hessian}\\'
        #     r'\lambda_1 > 0,\ \lambda_2 < 0\\'
        #     r'\text{Saddle point}',
        #     font_size=18,
        #     color=ORANGE
        # )
        # label3.move_to(RIGHT * 3 + DOWN * 1.8)
        
        # self.play(Write(label1))
        # self.play(Write(label2))
        # self.play(Write(label3))
        
        # 添加底部说明
        # explanation = MathTex(
        #     r'\text{Curvature is determined by the eigenvalues of the Hessian:}\\'
        #     r'\lambda > 0 \rightarrow \text{positive curvature (convex up)}\\'
        #     r'\lambda < 0 \rightarrow \text{negative curvature (concave down)}',
        #     font_size=18,
        #     color=WHITE
        # )
        # explanation.to_edge(DOWN, buff=0.5)
        # self.play(Write(explanation))
        
        # self.wait(2)
        
        # 导出静态图片
        # self.renderer.update_frame()
        # self.renderer.get_image().save(
        #     'hessian_curvature_combined.png'
        # )
        print("已保存: hessian_curvature_combined.png")


class EigenvalueVisualizationScene(ThreeDScene):
    """
    特征值与曲率方向可视化
    
    展示Hessian矩阵的特征值如何决定曲面在不同方向上的曲率
    """
    
    def construct(self):
        """构建场景"""
        
        # 设置相机视角
        self.set_camera_orientation(
            phi=60 * DEGREES,
            theta=35 * DEGREES,
            focal_distance=2
        )
        
        # 标题
        title = Text(
            'Eigenvalues and Curvature Directions\n'
            r'$\mathbf{Hv} = \lambda \mathbf{v}$',
            font_size=32,
            color=BLUE
        )
        title.to_edge(UP, buff=0.5)
        self.add(title)
        
        # 创建鞍面（最能展示不同方向曲率的曲面）
        surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                saddle_surface(u, v) / 2
            ]),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(35, 35)
        )
        surface.set_style(
            fill_opacity=0.75,
            stroke_color=WHITE,
            stroke_width=0.4
        )
        surface.set_fill_by_value(
            axes=saddle_surface(np.meshgrid(np.linspace(-2, 2, 35), 
                                          np.linspace(-2, 2, 35))[0],
                              np.meshgrid(np.linspace(-2, 2, 35),
                                          np.linspace(-2, 2, 35))[1]) / 2,
            cmap='RdYlBu_r',
            max=2,
            min=-2
        )
        
        self.play(Create(surface), run_time=2)
        
        # 绘制特征向量方向
        # λ₁ = 2 对应x方向（正曲率）
        arrow_x = Arrow3D(
            start=np.array([-0.5, 0, -0.2]),
            end=np.array([0.5, 0, 0.2]),
            color=RED,
            stroke_width=4
        )
        
        # λ₂ = -2 对应y方向（负曲率）
        arrow_y = Arrow3D(
            start=np.array([0, -0.5, 0.2]),
            end=np.array([0, 0.5, -0.2]),
            color=BLUE,
            stroke_width=4
        )
        
        self.add(arrow_x, arrow_y)
        
        # 特征向量标签
        label_x = Text(r'v₁ (λ=2)', font_size=20, color=RED)
        label_x.move_to([0.8, 0, 0.5])
        
        label_y = Text(r'v₂ (λ=-2)', font_size=20, color=BLUE)
        label_y.move_to([0, 0.8, -0.3])
        
        self.add(label_x, label_y)
        
        # 添加公式
        formula = MathTex(
            r'\mathbf{H} = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}'
            r'\quad\Rightarrow\quad'
            r'\begin{cases} \mathbf{Hv}_1 = 2\mathbf{v}_1 & (\text{正曲率})\\'
            r'\\ \mathbf{Hv}_2 = -2\mathbf{v}_2 & (\text{负曲率}) \end{cases}',
            font_size=20,
            color=YELLOW
        )
        formula.to_edge(DOWN, buff=0.8)
        
        self.play(Write(formula))
        
        self.wait(2)
        
        # 导出静态图片
        # self.renderer.update_frame()
        # self.renderer.get_image().save(
        #     'hessian_eigenvalue_directions.png'
        # )
        frame = self.take_frame()
        frame.save('hessian_positive_definite.png')
        print("已保存: hessian_eigenvalue_directions.png")


# ============================================================================
# 主程序入口
# ============================================================================

def render_all_scenes():
    """
    渲染所有场景并生成静态图片
    
    注意：使用manim命令渲染效果更好
    """
    
    print("=" * 60)
    print("Hessian矩阵与曲率关系可视化")
    print("Hessian Matrix and Curvature Visualization")
    print("=" * 60)
    print()
    print("使用manim命令渲染:")
    print("  manim -pqh hessian_curvature_manim.py PositiveHessianScene")
    print("  manim -pqh hessian_curvature_manim.py NegativeHessianScene")
    print("  manim -pqh hessian_curvature_manim.py SaddlePointScene")
    print("  manim -pqh hessian_curvature_manim.py CombinedScene")
    print("  manim -pqh hessian_curvature_manim.py EigenvalueVisualizationScene")
    print()
    print("参数说明:")
    print("  -p: 预览")
    print("  -q: 质量 (l=低, m=中, h=高)")
    print("  -h: 高质量")
    print()
    print("或在代码中直接运行（使用renderer）")
    print("=" * 60)
    
    # 打印数学信息
    print("\n数学理论总结:")
    print("-" * 40)
    print("1. 正定Hessian (λ₁>0, λ₂>0):")
    print("   → 局部极小值点（碗状曲面）")
    print()
    print("2. 负定Hessian (λ₁<0, λ₂<0):")
    print("   → 局部极大值点（倒碗状曲面）")
    print()
    print("3. 混合Hessian (λ₁>0, λ₂<0):")
    print("   → 鞍点（马鞍面）")
    print()
    print("4. 半正定/半负定:")
    print("   → 需要进一步分析")
    print("-" * 40)


if __name__ == "__main__":
    render_all_scenes()
