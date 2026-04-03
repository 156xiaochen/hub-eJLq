"""
使用 CLIP 模型进行 Zero-Shot 图像分类
"""
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import os

# 配置
MODEL_PATH = "D:/Python/study/badou/models/chinese-clip-vit-base-patch16"
IMAGE_PATH = "D:/Python/study/badou/Week10/assets/test_images/dog.jpg"  # 请替换为你的小狗图片路径

# 候选标签（中文）
CANDIDATE_LABELS = [
    "狗",
    "猫", 
    "鸟",
    "鱼",
    "兔子",
    "仓鼠",
    "狐狸",
    "狼"
]

def load_model_and_processor(model_path):
    """加载 CLIP 模型和处理器"""
    print("正在加载模型...")
    # 使用 AutoModel 和 AutoProcessor 来正确加载 Chinese-CLIP
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

def classify_image(model, processor, image_path, candidate_labels):
    """对图像进行 zero-shot 分类"""
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 准备输入
    inputs = processor(
        text=candidate_labels,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # 图像 - 文本相似度分数
        
        # 计算概率
        probs = logits_per_image.softmax(dim=1)
    
    return candidate_labels, probs

def main():
    """主函数"""
    # 检查图片是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"错误：图片不存在于 {IMAGE_PATH}")
        print("请修改 IMAGE_PATH 变量指向你的小狗图片")
        return
    
    # 加载模型
    model, processor = load_model_and_processor(MODEL_PATH)
    
    # 进行分类
    print(f"\n正在对图像进行分类：{IMAGE_PATH}")
    labels, probs = classify_image(model, processor, IMAGE_PATH, CANDIDATE_LABELS)
    
    # 输出结果
    print("\n=== 分类结果 ===")
    for label, prob in zip(labels, probs[0]):
        print(f"{label}: {prob:.4f} ({prob*100:.2f}%)")
    
    # 获取最可能的预测
    top_idx = probs.argmax()
    print(f"\n预测结果：{labels[top_idx]} (置信度：{probs[0][top_idx]*100:.2f}%)")

if __name__ == "__main__":
    main()
