import re
import os

def process_latex_file(input_path, output_path):
    """
    读取 LaTeX 文件，将自定义图片格式转换为自适应宽度的标准 LaTeX 格式
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误：在路径 '{input_path}' 未找到文件。")
        return

    # 正则表达式说明：
    # 匹配 !{[}{[}文件名{]}{]}
    pattern = r'!\{\[\}\{\[\}(.*?)\{\]\}\{\]\}'
    
    # 替换格式说明：
    # [width=\\textwidth]: 确保图片宽度自适应页面，不超出边界
    # ../graph/\1: 使用你指定的相对路径
    replacement = r'\\includegraphics[width=\\textwidth]{../graph/\1}'

    try:
        # 1. 读取原始文件内容
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. 执行正则替换
        # count 可以记录替换了多少处
        new_content, count = re.subn(pattern, replacement, content)

        # 3. 写入处理后的内容
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ 处理成功！")
        print(f"📝 共识别并转换了 {count} 张图片。")
        print(f"💾 结果保存至: {output_path}")

    except Exception as e:
        print(f"💥 运行中出现错误: {e}")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 假设你的目录结构是：
    # 项目根目录/
    # ├── latex_file/
    # │   └── output.tex
    # └── graph/
    #     └── (图片文件)
    
    BASE_DIR = "latex_file"
    INPUT_NAME = "output.tex"
    OUTPUT_NAME = "output_fixed.tex"
    
    input_file_path = os.path.join(BASE_DIR, INPUT_NAME)
    output_file_path = os.path.join(BASE_DIR, OUTPUT_NAME)

    # 执行转换
    process_latex_file(input_file_path, output_file_path)