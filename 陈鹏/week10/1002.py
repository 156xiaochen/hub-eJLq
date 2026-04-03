from openai import OpenAI
import fitz  # PyMuPDF
import base64

"""
使用云端的Qwen-VL 对本地的pdf（任意pdf的第一页） 进行解析，写一下这个代码
"""

# 初始化OpenAI客户端
client = OpenAI(
    # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
    api_key="sk-fe0209453f0d48179de8bd53a6ce028c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def pdf_to_images(pdf_path: str, page_number: int = 1, dpi: int = 300) -> str:
    """
    将本地 PDF 的指定页转换为 Base64 编码的图片字符串。
    """
    if page_number < 1:
        raise ValueError("页码必须从 1 开始")

    doc = None
    try:
        # 打开 PDF
        doc = fitz.open(pdf_path)
        # 获取指定页 (页码从 0 开始，所以要 -1)
        page = doc[page_number - 1]
        # 设置缩放矩阵 (dpi / 72.0 是 PDF 的标准缩放公式)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        # 渲染页面为图片 (pixmap)
        pix = page.get_pixmap(matrix=mat)
        # 转换为 JPEG 字节
        img_bytes = pix.tobytes("jpeg")
        # 转 Base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"

    except Exception as e:
        print(f"❌ PyMuPDF 转换失败: {e}")
        raise e
    finally:
        # 确保文件句柄被释放
        if doc is not None:
            doc.close()


pdf_path = "测试.pdf"  # 你的PDF路径
image_base64_data = pdf_to_images(pdf_path, page_number=1, dpi=300)

# 创建聊天完成请求
response = client.chat.completions.create(
    model='qwen-vl-max-latest',
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_base64_data}},  # 标准格式
                {"type": "text", "text": "请详细识别图片中的所有文字，准确提取内容并总结文档核心信息"}
            ]
        }
    ]
)

print("=" * 50)
print("PDF 第一页解析结果：")
print("=" * 50)
print(response.choices[0].message.content)
