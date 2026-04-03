"""
使用阿里云 Qwen-VL 解析 PDF 第一页
功能：将本地 PDF 第一页转换为图片，然后调用云端 Qwen-VL 进行解析
"""

import base64
from pathlib import Path
from openai import OpenAI
import fitz  # PyMuPDF


def pdf_to_base64(pdf_path: str, page_num: int = 0) -> str:
    """
    将 PDF 指定页面转换为 base64 编码的图片
    
    Args:
        pdf_path: PDF 文件路径
        page_num: 页码（从 0 开始，默认第一页）
    
    Returns:
        base64 编码的图片字符串
    """
    # 打开 PDF 文档
    doc = fitz.open(pdf_path)
    
    # 检查页码是否有效
    if page_num >= len(doc):
        raise ValueError(f"页码超出范围，PDF 共有 {len(doc)} 页")
    
    # 获取指定页面
    page = doc[page_num]
    
    # 设置缩放比例（300 DPI，提高清晰度）
    zoom = 300 / 72  # 72 是 PDF 的默认 DPI
    mat = fitz.Matrix(zoom, zoom)
    
    # 渲染页面为图片
    pix = page.get_pixmap(matrix=mat)
    
    # 转换为 PNG 格式的 bytes
    img_bytes = pix.tobytes("png")
    
    # 关闭文档
    doc.close()
    
    # 转换为 base64
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    
    return base64_str


def analyze_pdf_with_qwen_vl(
    pdf_path: str,
    api_key: str,
    prompt: str = "请详细描述这张图片的内容",
    page_num: int = 0,
    model: str = "qwen-vl-max-latest"
) -> str:
    """
    使用 Qwen-VL 分析 PDF 页面内容
    
    Args:
        pdf_path: PDF 文件路径
        api_key: 阿里云百炼 API Key
        prompt: 提示词
        page_num: 页码（从 0 开始）
        model: 模型名称，可选值：
               - qwen-vl-max-latest (推荐)
               - qwen-vl-plus-latest
               - qwen-vl-ocr-latest
    
    Returns:
        模型返回的分析结果
    """
    # 验证文件是否存在
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")
    
    print(f"正在处理 PDF: {pdf_path}")
    print(f"提取第 {page_num + 1} 页...")
    
    # 将 PDF 页面转为 base64
    base64_image = pdf_to_base64(pdf_path, page_num)
    
    # 创建 OpenAI 客户端（阿里云百炼兼容 OpenAI 格式）
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    print(f"正在调用 {model} 模型进行分析...")
    
    # 构造请求
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "你是一个专业的文档分析助手。"}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    # 获取返回结果
    result = completion.choices[0].message.content
    
    return result


def main():
    """主函数示例"""
    # ========== 配置参数 ==========
    # 替换为你的阿里云百炼 API Key
    # 获取方式：登录 https://bailian.console.aliyun.com/ 
    # -> 创建 API Key -> 复制保存
    API_KEY = "sk-be4235589ac240b099ce67bc1af07581"
    
    # PDF 文件路径
    PDF_PATH = r"D:\Python\study\badou\Week07\2404.16130v2-GraphRAG.pdf"  # 替换为你的 PDF 路径
    
    # 分析提示词（可根据需求修改）
    PROMPT = """
请详细分析这份文档的第一页，包括：
1. 文档类型和主题
2. 主要内容概述
3. 关键信息提取
4. 如果有表格或图表，请描述其内容
"""
    
    # ========== 执行分析 ==========
    try:
        result = analyze_pdf_with_qwen_vl(
            pdf_path=PDF_PATH,
            api_key=API_KEY,
            prompt=PROMPT,
            page_num=0,  # 第一页（页码从 0 开始）
            model="qwen-vl-max-latest"  # 使用最强版本
        )
        
        print("\n" + "=" * 60)
        print("【分析结果】")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"❌ 错误：{e}")
    except Exception as e:
        print(f"❌ 分析失败：{e}")
        print("\n请检查:")
        print("1. PDF 文件路径是否正确")
        print("2. API Key 是否有效")
        print("3. 网络连接是否正常")


if __name__ == "__main__":
    main()
