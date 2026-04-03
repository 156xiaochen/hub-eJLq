import os
import base64
import fitz  # pymupdf
from dashscope import MultiModalConversation
import dashscope
from oss_uploader import OSSUploader  # 导入改造后的类

# === 配置 ===
pdf_path = "test.pdf"
api_key = "sk-1f8f970c557d41b9899269dc981366f9"

# 设置 DashScope API Key
dashscope.api_key = api_key

# === 1. 初始化OSS上传器（无需传参）===
oss_uploader = OSSUploader()  # 直接使用内置配置

# === 2. PDF转图片 ===
print("正在转换 PDF 为图片...")
doc = fitz.open(pdf_path)
page = doc[0]

zoom = 2.0
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat)

image_path = "temp_page.png"
pix.save(image_path)
print(f"PDF 转图片完成: {image_path}")

doc.close()

# === 3. 上传图片到OSS ===
print("\n正在上传图片到OSS...")
image_url = oss_uploader.upload_image(image_path, prefix="ocr_images")

if image_url is None:
    # 上传失败，直接抛出异常
    raise Exception("OSS上传失败，无法继续执行OCR识别")

print(f"图片URL: {image_url}")

# === 4. 调用OCR API ===
print("\n正在调用 OCR API...")
response = MultiModalConversation.call(
    model="qwen-vl-ocr",
    messages=[{
        'role': 'user',
        'content': [{'image': image_url}]
    }]
)

# === 5. 输出结果 ===
if response.status_code == 200:
    content = response.output.choices[0].message.content
    if isinstance(content, list):
        text = ""
        for item in content:
            if 'text' in item:
                text += item['text']
    else:
        text = str(content)

    print("\n" + "=" * 50)
    print("识别结果:")
    print("=" * 50)
    print(text)
else:
    print(f"API 调用失败，状态码: {response.status_code}")
    print(f"错误信息: {response.message}")

# === 6. 清理临时文件 ===
if os.path.exists(image_path):
    os.remove(image_path)
    print("\n临时文件已清理")
