"""
反向传播算法可视化 - 链式法则与误差信号传播
Backpropagation Visualization: Chain Rule & Error Signal Propagation

使用方法：
1. 安装 Manim: pip install manim
2. 运行渲染: manim -pql backpropagation_visualization.py BackpropagationScene
   - -p: 预览
   - -ql: 低质量（快速测试）
   - -qh: 高质量（最终输出）

作者：MiniMax Agent
"""

from manim import *

class BackpropagationScene(Scene):
    """反向传播可视化主场景"""
    
    def construct(self):
        # ==================== 配置部分 ====================
        self.setup_colors()
        self.setup_structure()
        
        # ==================== 场景1：初始化神经网络 ====================
        self.introduce_network()
        
        # ==================== 场景2：展示误差传播公式 ====================
        self.show_propagation_formula()
        
        # ==================== 场景3：输出层误差激活 ====================
        self.visualize_output_error()
        
        # ==================== 场景4：误差反向传播（权重矩阵转置） ====================
        self.visualize_weight_transpose()
        
        # ==================== 场景5：激活函数导数缩放 ====================
        self.visualize_activation_derivative()
        
        # ==================== 场景6：隐藏层误差信号 ====================
        self.show_hidden_error_signal()
        
        # ==================== 场景7：权重梯度计算 ====================
        self.visualize_weight_gradient()
        
        # ==================== 场景8：总结 ====================
        self.show_summary()
        
        # 结束
        self.wait(2)
    
    def setup_colors(self):
        """设置颜色方案"""
        # 网络层颜色
        self.COLOR_INPUT = BLUE          # 输入层 - 蓝色
        self.COLOR_HIDDEN = GREEN        # 隐藏层 - 绿色
        self.COLOR_OUTPUT = RED          # 输出层 - 红色
        
        # 计算过程颜色
        self.COLOR_ERROR = YELLOW        # 误差信号 - 黄色（高亮）
        self.COLOR_DERIVATIVE = PURPLE   # 激活函数导数 - 紫色
        self.COLOR_WEIGHT_GRAD = ORANGE  # 权重梯度 - 橙色
        self.COLOR_ACTIVATION = TEAL     # 激活值 - 青色
        
        # 其他
        self.COLOR_WEIGHTS = GREY        # 连接权重 - 灰色
        self.BG_COLOR = "#1a1a2e"        # 背景色 - 深蓝黑
    
    def setup_structure(self):
        """设置网络结构配置"""
        self.layer_sizes = [3, 4, 2]     # 每层节点数：[输入层, 隐藏层, 输出层]
        self.layer_spacing = 3.0         # 层间距
        self.node_spacing = 1.5          # 节点间距
        self.node_radius = 0.35          # 节点半径
        
        self.network_group = None
        self.layers = None
        self.edges_l1 = None  # 输入到隐藏的连接
        self.edges_l2 = None  # 隐藏到输出的连接
    
    def create_network(self):
        """创建神经网络图形"""
        layers = VGroup()
        edges_l1 = VGroup()  # 输入层 -> 隐藏层
        edges_l2 = VGroup()  # 隐藏层 -> 输出层
        
        # 创建各层节点
        for i, size in enumerate(self.layer_sizes):
            layer = VGroup()
            for j in range(size):
                # 计算位置：居中对齐
                y_pos = (j - (size - 1) / 2) * self.node_spacing
                x_pos = (i - 1) * self.layer_spacing  # 隐藏层在中心
                
                # 创建节点
                node = Circle(
                    radius=self.node_radius,
                    color=WHITE,
                    fill_opacity=0.9
                )
                
                # 设置节点颜色
                if i == 0:
                    node.set_fill(self.COLOR_INPUT)
                elif i == 1:
                    node.set_fill(self.COLOR_HIDDEN)
                else:
                    node.set_fill(self.COLOR_OUTPUT)
                
                node.move_to([x_pos, y_pos, 0])
                layer.add(node)
            layers.add(layer)
        
        # 创建连接边（权重）
        for n_input in layers[0]:
            for n_hidden in layers[1]:
                edge = Line(
                    n_input.get_center(),
                    n_hidden.get_center(),
                    color=self.COLOR_WEIGHTS,
                    stroke_width=2,
                    stroke_opacity=0.6
                )
                edges_l1.add(edge)
        
        for n_hidden in layers[1]:
            for n_output in layers[2]:
                edge = Line(
                    n_hidden.get_center(),
                    n_output.get_center(),
                    color=self.COLOR_WEIGHTS,
                    stroke_width=2,
                    stroke_opacity=0.6
                )
                edges_l2.add(edge)
        
        self.layers = layers
        self.edges_l1 = edges_l1
        self.edges_l2 = edges_l2
        self.network_group = VGroup(edges_l1, edges_l2, layers)
    
    def create_layer_labels(self):
        """创建层标签"""
        labels = VGroup()
        
        # 输入层标签
        label_input = Text("输入层\nx", font_size=18, color=self.COLOR_INPUT)
        label_input.next_to(self.layers[0], UP, buff=0.5)
        labels.add(label_input)
        
        # 隐藏层标签
        label_hidden = Text("隐藏层 (ℓ)\nh", font_size=18, color=self.COLOR_HIDDEN)
        label_hidden.next_to(self.layers[1], UP, buff=0.5)
        labels.add(label_hidden)
        
        # 输出层标签
        label_output = Text("输出层 (ℓ+1)\nŷ", font_size=18, color=self.COLOR_OUTPUT)
        label_output.next_to(self.layers[2], UP, buff=0.5)
        labels.add(label_output)
        
        return labels
    
    def introduce_network(self):
        """场景1：介绍神经网络结构"""
        self.camera.background_color = self.BG_COLOR
        
        # 创建网络
        self.create_network()
        
        # 渐入网络
        self.play(
            FadeIn(self.network_group, shift=UP),
            run_time=1.5
        )
        
        # 添加标签
        labels = self.create_layer_labels()
        self.play(Write(labels), run_time=1)
        
        # 脉冲动画突出显示网络结构
        self.play(
            self.network_group.animate.scale(1.05),
            run_time=0.5,
            rate_func=there_and_back
        )
        
        self.wait(1)
        
        # 存储标签以便后续使用
        self.network_labels = labels
    
    def show_propagation_formula(self):
        """场景2：展示误差传播公式"""
        # 标题
        title = Text("反向传播核心原理", font_size=28, color=self.COLOR_ERROR)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title), run_time=1)
        
        # 主公式：δ(l) = (W(l+1))^T · δ(l+1) ⊙ σ'(z(l))
        main_formula = MathTex(
            r"\boldsymbol{\delta}^{(\ell)}",
            r"=",
            r"\left(\mathbf{W}^{(\ell+1)}\right)^T",
            r"\boldsymbol{\delta}^{(\ell+1)}",
            r"\odot",
            r"\phi'\left(\mathbf{z}^{(\ell)}\right)",
            font_size=32
        )
        
        # 设置公式各部分颜色
        main_formula[0].set_color(self.COLOR_ERROR)      # δ(l) - 当前层误差
        main_formula[3].set_color(self.COLOR_OUTPUT)     # δ(l+1) - 下一层误差
        main_formula[2].set_color(BLUE_B)                # W^T - 权重转置
        main_formula[5].set_color(self.COLOR_DERIVATIVE) # σ' - 激活导数
        
        # 放置在底部
        main_formula.next_to(self.network_group, DOWN, buff=0.8)
        
        self.play(Write(main_formula), run_time=2)
        self.wait(1)
        
        # 公式说明
        explanation = VGroup()
        
        line1 = Text("• 误差信号从输出层反向传播到输入层", font_size=16, color=WHITE)
        line2 = Text("• 权重矩阵转置：反向路由误差到对应维度", font_size=16, color=WHITE)
        line3 = Text("• 激活函数导数：缩放误差信号", font_size=16, color=WHITE)
        
        explanation.add(line1, line2, line3)
        explanation.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        explanation.to_corner(DR, buff=0.5)
        
        self.play(FadeIn(explanation, shift=LEFT), run_time=1.5)
        
        self.wait(1.5)
        
        # 保存公式引用
        self.main_formula = main_formula
        self.formula_explanation = explanation
    
    def visualize_output_error(self):
        """场景3：可视化输出层误差"""
        # 高亮输出层 - 显示误差 δ(l+1)
        self.play(
            LaggedStart(
                *[Indicate(node, color=self.COLOR_ERROR, scale_factor=1.3) 
                  for node in self.layers[2]],
                lag_ratio=0.15
            ),
            run_time=1.5
        )
        
        # 在输出层旁边显示误差符号
        error_labels = VGroup()
        for i, node in enumerate(self.layers[2]):
            delta_label = MathTex(f"\\delta^{{(L)}}_{{{i}}}", font_size=20, color=self.COLOR_ERROR)
            delta_label.next_to(node, RIGHT, buff=0.3)
            error_labels.add(delta_label)
        
        self.play(Write(error_labels), run_time=1)
        
        # 脉冲效果
        self.play(
            self.layers[2].animate.set_fill(self.COLOR_ERROR),
            run_time=0.5
        )
        self.play(
            self.layers[2].animate.set_fill(self.COLOR_OUTPUT),
            run_time=0.5
        )
        
        self.wait(1)
        
        # 高亮公式中的相关部分
        self.play(
            Indicate(self.main_formula[3], color=self.COLOR_OUTPUT, scale_factor=1.2),
            run_time=0.8
        )
        
        self.wait(0.5)
        
        self.output_error_labels = error_labels
    
    def visualize_weight_transpose(self):
        """场景4：可视化权重矩阵转置（误差反向传播）"""
        # 高亮公式中的权重部分
        self.play(
            Indicate(self.main_formula[2], color=BLUE_B, scale_factor=1.2),
            run_time=0.8
        )
        
        # 激活连接线 - 显示它们正在被"反向"使用
        self.play(
            self.edges_l2.animate.set_color(BLUE_B).set_stroke(width=4, opacity=1),
            run_time=0.5
        )
        
        # 创建反向传播的粒子动画
        particles = VGroup()
        particle_paths = []
        
        # 为每个连接创建反向路径
        for n_output in self.layers[2]:
            for n_hidden in self.layers[1]:
                # 在输出节点创建粒子
                particle = Dot(
                    color=self.COLOR_ERROR,
                    radius=0.1,
                    fill_opacity=0.9
                )
                particle.move_to(n_output.get_center())
                particles.add(particle)
                
                # 创建从输出到隐藏的路径
                path = Line(
                    n_output.get_center(),
                    n_hidden.get_center(),
                    color=self.COLOR_ERROR,
                    stroke_width=3,
                    stroke_opacity=0.8
                )
                particle_paths.append((particle, path))
        
        # 渐入粒子
        self.play(
            FadeIn(particles, scale=1.5),
            run_time=0.5
        )
        
        # 粒子沿路径反向移动（从输出到隐藏）
        # for particle, path in particle_paths:
            # self.play(
            #     MoveAlongPath(particle, path, run_time=1.5, rate_func=linear),
            #     run_time=0
            # )  # 收集动画
        self.play(
            *[
                MoveAlongPath(particle, path, run_time=1.5, rate_func=linear)
                for particle, path in particle_paths
            ]
        )
        # 执行所有粒子动画
        self.play(
            *[
                MoveAlongPath(particle, path, run_time=1.5, rate_func=linear)
                for particle, path in particle_paths
            ]
        )
        
        # 移除粒子
        self.play(FadeOut(particles), run_time=0.3)
        
        # 重置连接线颜色
        self.play(
            self.edges_l2.animate.set_color(self.COLOR_WEIGHTS).set_stroke(width=2, opacity=0.6),
            run_time=0.5
        )
        
        self.wait(1)
    
    def visualize_activation_derivative(self):
        """场景5：可视化激活函数导数的缩放作用"""
        # 高亮公式中的导数部分
        self.play(
            Indicate(self.main_formula[5], color=self.COLOR_DERIVATIVE, scale_factor=1.2),
            run_time=0.8
        )
        
        # 在隐藏层旁边显示激活导数
        derivative_labels = VGroup()
        for i, node in enumerate(self.layers[1]):
            sigma_label = MathTex(f"\\phi'_{{{i}}}", font_size=18, color=self.COLOR_DERIVATIVE)
            sigma_label.next_to(node, LEFT, buff=0.3)
            derivative_labels.add(sigma_label)
        
        self.play(Write(derivative_labels), run_time=1)
        
        # 闪烁隐藏层节点 - 表示激活导数的作用
        flash_animations = []
        for node in self.layers[1]:
            flash = Flash(
                node,
                color=self.COLOR_DERIVATIVE,
                line_length=0.3,
                num_lines=10,
                # stroke_width=2
            )
            flash_animations.append(flash)
        
        # 改变节点颜色表示导数缩放
        self.play(
            self.layers[1].animate.set_fill(self.COLOR_DERIVATIVE),
            run_time=0.5
        )
        
        self.play(
            AnimationGroup(*flash_animations, group=self.layers[1]),
            run_time=1
        )
        
        # 恢复节点颜色（但保持某种变化表示已经处理）
        self.play(
            self.layers[1].animate.set_fill(self.COLOR_HIDDEN),
            run_time=0.5
        )
        
        self.wait(1)
        
        self.derivative_labels = derivative_labels
    
    def show_hidden_error_signal(self):
        """场景6：显示隐藏层的误差信号"""
        # 公式中的等号左边高亮
        self.play(
            Indicate(self.main_formula[0], color=self.COLOR_ERROR, scale_factor=1.2),
            run_time=0.8
        )
        
        # 隐藏层节点变为黄色 - 表示现在是误差 δ(l)
        self.play(
            self.layers[1].animate.set_fill(self.COLOR_ERROR),
            run_time=0.5
        )
        
        # 在隐藏层旁边显示误差符号
        hidden_error_labels = VGroup()
        for i, node in enumerate(self.layers[1]):
            delta_label = MathTex(r"\delta^{(\ell)}_{" + str(i) + r"}", font_size=20, color=self.COLOR_ERROR)
            delta_label.next_to(node, LEFT, buff=0.3)
            hidden_error_labels.add(delta_label)
        
        self.play(
            FadeOut(self.derivative_labels),
            Write(hidden_error_labels),
            run_time=1
        )
        
        # 输出层误差标签淡出
        self.play(FadeOut(self.output_error_labels), run_time=0.5)
        
        # 脉冲效果确认
        self.play(
            self.layers[1].animate.scale(1.1),
            run_time=0.3,
            rate_func=there_and_back
        )
        
        self.wait(1)
        
        self.hidden_error_labels = hidden_error_labels
    
    def visualize_weight_gradient(self):
        """场景7：可视化权重梯度计算"""
        # 隐藏公式，显示梯度公式
        self.play(
            FadeOut(self.main_formula),
            FadeOut(self.formula_explanation),
            run_time=0.5
        )
        
        # 新的标题
        new_title = Text("权重梯度计算", font_size=24, color=self.COLOR_WEIGHT_GRAD)
        new_title.to_edge(UP, buff=0.5)
        
        self.play(
            Transform(self.network_labels[1], new_title),
            run_time=0.5
        )
        
        # 梯度公式：∂L/∂W = δ(l) · (h(l-1))^T
        gradient_formula = MathTex(
            r"\frac{\partial L}{\partial \mathbf{W}^{(\ell)}}",
            r"=",
            r"\boldsymbol{\delta}^{(\ell)}",
            r"\left(\mathbf{h}^{(\ell-1)}\right)^T",
            font_size=28
        )
        
        # 设置颜色
        gradient_formula[0].set_color(self.COLOR_WEIGHT_GRAD)  # 梯度
        gradient_formula[2].set_color(self.COLOR_ERROR)        # 误差
        gradient_formula[3].set_color(self.COLOR_ACTIVATION)   # 激活值
        
        gradient_formula.next_to(self.network_group, DOWN, buff=0.8)
        
        self.play(Write(gradient_formula), run_time=1.5)
        
        # 动画：隐藏层误差（黄色） + 输入层激活（蓝色） -> 权重梯度（橙色）
        
        # 1. 高亮隐藏层误差
        self.play(
            Indicate(self.hidden_error_labels, color=self.COLOR_ERROR, scale_factor=1.2),
            self.layers[1].animate.scale(1.15).set_fill(self.COLOR_ERROR),
            run_time=0.8
        )
        
        # 2. 高亮输入层激活
        self.play(
            self.layers[0].animate.scale(1.15).set_fill(self.COLOR_ACTIVATION),
            run_time=0.8
        )
        
        # 3. 连接线变为橙色 - 表示梯度
        self.play(
            self.edges_l1.animate.set_color(self.COLOR_WEIGHT_GRAD).set_stroke(width=4, opacity=1),
            run_time=0.8
        )
        
        # 4. 公式脉冲
        self.play(
            Wiggle(gradient_formula[0], run_time=1)
        )
        
        # 恢复
        self.play(
            self.layers[0].animate.scale(1/1.15).set_fill(self.COLOR_INPUT),
            self.layers[1].animate.scale(1/1.15).set_fill(self.COLOR_HIDDEN),
            run_time=0.5
        )
        
        # 显示权重梯度符号
        grad_labels = VGroup()
        for edge in self.edges_l1:
            grad_label = MathTex(
                r"\frac{\partial L}{\partial w}",
                font_size=12,
                color=self.COLOR_WEIGHT_GRAD
            )
            grad_label.move_to(edge.get_center())
            grad_labels.add(grad_label)
        
        self.play(Write(grad_labels), run_time=1)
        
        self.wait(1.5)
        
        # 清理
        self.play(
            FadeOut(gradient_formula),
            FadeOut(grad_labels),
            run_time=0.5
        )
        
        self.weight_grad_labels = grad_labels
    
    def show_summary(self):
        """场景8：总结"""
        # 清理当前元素
        self.play(
            FadeOut(self.hidden_error_labels),
            self.edges_l1.animate.set_color(self.COLOR_WEIGHTS).set_stroke(width=2, opacity=0.6),
            run_time=0.5
        )
        
        # 恢复网络颜色
        self.play(
            self.layers[0].animate.set_fill(self.COLOR_INPUT),
            self.layers[1].animate.set_fill(self.COLOR_HIDDEN),
            self.layers[2].animate.set_fill(self.COLOR_OUTPUT),
            run_time=0.5
        )
        
        # 总结标题
        summary_title = Text("反向传播总结", font_size=32, color=self.COLOR_ERROR)
        summary_title.to_edge(UP, buff=0.5)
        
        self.play(
            Transform(self.network_labels[1], summary_title),
            run_time=0.5
        )
        
        # 总结要点
        summary_points = VGroup()
        
        point1 = Text("1. 误差信号从输出层传播到输入层", font_size=20, color=WHITE)
        point2 = Text("2. 权重矩阵转置实现误差的反向路由", font_size=20, color=WHITE)
        point3 = Text("3. 激活函数导数缩放误差信号", font_size=20, color=WHITE)
        point4 = Text("4. 权重梯度 = 误差信号 × 前一层激活值", font_size=20, color=WHITE)
        
        summary_points.add(point1, point2, point3, point4)
        summary_points.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        summary_points.to_corner(DR, buff=0.5)
        
        self.play(Write(summary_points), run_time=2)
        
        # 最终脉冲动画
        self.play(
            self.network_group.animate.scale(1.08),
            run_time=0.5,
            rate_func=there_and_back
        )
        
        self.wait(2)
        
        # 淡出
        self.play(
            FadeOut(summary_points),
            FadeOut(summary_title),
            run_time=1
        )


# ==================== 额外场景：偏置梯度 ====================

class BiasGradientScene(Scene):
    """偏置梯度可视化场景"""
    
    def construct(self):
        self.setup_colors()
        self.setup_structure()
        
        # 创建简化的网络（更聚焦于偏置）
        self.create_network()
        self.play(FadeIn(self.network_group), run_time=1)
        
        # 偏置梯度公式
        bias_grad_formula = MathTex(
            r"\frac{\partial L}{\partial \mathbf{b}^{(\ell)}}",
            r"=",
            r"\boldsymbol{\delta}^{(\ell)}",
            font_size=36
        )
        
        bias_grad_formula[0].set_color(PINK)
        bias_grad_formula[2].set_color(YELLOW)
        
        bias_grad_formula.to_edge(DOWN, buff=1)
        
        self.play(Write(bias_grad_formula), run_time=1.5)
        
        # 显示偏置
        biases = VGroup()
        for i, layer in enumerate(self.layers):
            for j, node in enumerate(layer):
                bias = Circle(
                    radius=0.15,
                    color=PINK,
                    fill_opacity=0.9
                )
                bias.move_to(node.get_center() + DOWN * 0.8)
                biases.add(bias)
        
        self.play(Write(biases), run_time=1)
        
        # 偏置 = 误差信号
        for i, layer in enumerate(self.layers):
            if i > 0:  # 隐藏层和输出层有偏置
                self.play(
                    Indicate(layer, color=YELLOW, scale_factor=1.2),
                    Indicate(biases[i*len(layer):(i+1)*len(layer)], color=PINK, scale_factor=1.2),
                    run_time=0.8
                )
        
        self.wait(2)


# ==================== 场景运行配置 ====================

if __name__ == "__main__":
    # 运行主场景
    print("请使用以下命令渲染动画：")
    print("manim -pql backpropagation_visualization.py BackpropagationScene")
    print("\n质量选项：")
    print("- -ql: 低质量（快速测试）")
    print("- -qm: 中等质量")
    print("- -qh: 高质量（最终输出）")
