import os
from pdf2image import convert_from_path
import dashscope
from dashscope import MultiModalConversation

dashscope.api_key = "sk-abu847ab13a543798c4860e15d459293"

pdf_path = "./example.pdf"

model_name = "qwen-vl-max"


def parse_pdf_with_qwen(pdf_file, api_key=None):
    if api_key:
        dashscope.api_key = api_key


    print(f"正在读取 PDF: {pdf_file} ...")
    try:
        images = convert_from_path(pdf_file, first_page=1, last_page=1, dpi=200)
        image = images[0]

        temp_image_path = "temp_page_1.jpg"
        image.save(temp_image_path, 'JPEG')
        print("PDF 第一页已转换为图片。")

    except Exception as e:
        print(f"PDF 转换失败，请检查是否安装了 poppler 且路径配置正确: {e}")
        return

    print(f"正在调用 {model_name} 进行解析...")

    messages = [
        {
            "role": "user",
            "content": [
                {"image": temp_image_path},
                {"text": "请详细解析这张图片中的内容。如果是文档，请提取标题、作者和摘要；如果是论文，请总结主要贡献。"}
            ]
        }
    ]

    try:
        response = MultiModalConversation.call(model=model_name, messages=messages)

        if response.status_code == 200:
            result_text = response.output.choices[0].message.content[0]['text']
            print("\n" + "=" * 30)
            print("Qwen-VL 解析结果:")
            print("=" * 30)
            print(result_text)
            print("=" * 30)
        else:
            print(f"调用失败: {response.code} - {response.message}")
            print(response.output)

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 清理临时文件
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print("\n临时文件已清理。")


if __name__ == "__main__":
    # 运行解析
    parse_pdf_with_qwen(pdf_path)