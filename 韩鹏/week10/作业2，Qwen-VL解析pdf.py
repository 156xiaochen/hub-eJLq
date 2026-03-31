import base64
from openai import OpenAI
from pdf2image import convert_from_path
import os

# ===================== 阿里云百炼客户端 =====================
client = OpenAI(
    api_key="",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ===================== PDF转图片 =====================
def pdf_to_images(pdf_path, poppler_path=None):
    if os.name == "nt" and not poppler_path:
        poppler_path = r"./poppler-25.12.0/Library/bin"
    return convert_from_path(pdf_path, poppler_path=poppler_path, dpi=150)


# ===================== 图片转base64 =====================
def image_to_base64(image):
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ===================== 主程序 =====================
if __name__ == "__main__":
    pdf_file = "./test.pdf"  # 本地PDF
    pdf_images = pdf_to_images(pdf_file)
    all_text = []

    for idx, img in enumerate(pdf_images):
        print(f"\n正在解析第 {idx + 1} 页...")

        # 调用云端Qwen-VL
        response = client.chat.completions.create(
            model="qwen-vl-plus",  # 官方云端多模态模型
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(img)}"}},
                    {"type": "text", "text": "提取图片中所有文字，完整输出"}
                ]
            }]
        )
        result = response.choices[0].message.content
        all_text.append(f"========== 第{idx + 1}页 ==========\n{result}")
        print(result)

    # 保存结果
    with open("pdf解析结果.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))

    print("\n 解析完成！")