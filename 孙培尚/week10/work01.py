"""
CLIP Zero-Shot Image Classification with Template Ensemble
使用本地 CLIP 模型进行图像分类，自动检测并使用 GPU（如果可用）
使用多种提示模板来提升分类准确率
"""
import os
# 屏蔽 oneDNN 警告
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# ========== 配置区域 ==========
# 设置本地模型路径
MODEL_PATH = r"D:\develop\badou\dataset\modelscope\models\AI-ModelScope\chinese-clip-vit-base-patch16"

# 图片路径（与脚本同目录下的 dog.png）
script_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(script_dir, "dog.png")

# 基础类别（纯类别名称）
BASE_CATEGORIES = [
    "狗",
    "猫",
    "汽车",
    "房子",
    "动物"
]

# 中文提示模板
TEMPLATES_ZH = [
    "一张{}的照片",
    "一张模糊的{}的照片",
    "一张裁剪过的{}的照片",
    "一张清晰的{}的照片",
    "一张{}的特写照片",
    "一张质量差的{}的照片",
    "一幅{}的画",
    "一幅{}的绘画",
    "一张高分辨率的{}的照片",
    "一张低分辨率的{}的照片"
]

# ========== 辅助函数 ==========
def generate_text_prompts(categories, templates):
    """
    生成所有提示文本
    Args:
        categories: 基础类别列表
        templates: 提示模板列表
    Returns:
        prompts: 生成的提示文本列表
        prompt_to_category: 每个提示对应的原始类别
    """
    prompts = []
    prompt_to_category = []

    for category in categories:
        for template in templates:
            prompt = template.format(category)
            prompts.append(prompt)
            prompt_to_category.append(category)

    return prompts, prompt_to_category

def ensemble_prediction(logits_per_image, prompt_to_category, categories):
    """
    对多个提示的结果进行集成投票
    Args:
        logits_per_image: 每个提示的logits [1, num_prompts]
        prompt_to_category: 每个提示对应的类别
        categories: 基础类别列表
    Returns:
        final_probs: 集成后的概率分布
    """
    # 计算每个提示的概率
    probs_per_prompt = F.softmax(logits_per_image, dim=1)  # [1, num_prompts]

    # 对每个类别的所有提示概率求和并平均
    category_probs = torch.zeros(1, len(categories)).to(probs_per_prompt.device)
    category_counts = torch.zeros(len(categories)).to(probs_per_prompt.device)

    for i, category in enumerate(prompt_to_category):
        category_idx = categories.index(category)
        category_probs[0, category_idx] += probs_per_prompt[0, i]
        category_counts[category_idx] += 1

    # 计算平均概率
    final_probs = category_probs / category_counts

    return final_probs

def main():
    print("=" * 60)
    print("CLIP Zero-Shot Image Classification with Template Ensemble")
    print("=" * 60)

    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 使用设备: {device}")
    if device == "cuda":
        print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")

    # 1. 加载本地模型
    print(f"\n正在加载本地模型：{MODEL_PATH}")
    try:
        model = CLIPModel.from_pretrained(MODEL_PATH).to(device)
        processor = CLIPProcessor.from_pretrained(MODEL_PATH)
        print("✓ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 读取图片
    print(f"\n正在读取图片：{IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 图片不存在: {IMAGE_PATH}")
        return

    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
        print(f"✓ 图片尺寸：{image.size}")
    except Exception as e:
        print(f"❌ 图片读取失败: {e}")
        return

    # 3. 生成提示文本
    print(f"\n生成提示模板...")
    print(f"基础类别数: {len(BASE_CATEGORIES)}")
    print(f"提示模板数: {len(TEMPLATES_ZH)}")

    all_prompts, prompt_to_category = generate_text_prompts(BASE_CATEGORIES, TEMPLATES_ZH)
    total_prompts = len(all_prompts)
    print(f"总提示数: {total_prompts}")

    print(f"\n示例提示:")
    for i in range(min(5, total_prompts)):
        print(f"  {i+1}. {all_prompts[i]} -> {prompt_to_category[i]}")

    # 4. 预处理输入（关键修复部分）
    print("\n正在进行预处理...")
    try:
        inputs = processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # ⚠️ 关键修复：移除 token_type_ids（中文 CLIP 不支持）
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
            print("✓ 已移除 token_type_ids")

        # 将张量移到对应设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print(f"✓ 预处理完成，文本数量: {len(all_prompts)}")
    except Exception as e:
        print(f"❌ 预处理失败: {e}")
        return

    # 5. 推理
    print("正在进行推理...")
    with torch.no_grad():
        outputs = model(**inputs)

    # 6. 集成预测
    logits_per_image = outputs.logits_per_image  # [1, total_prompts]
    final_probs = ensemble_prediction(logits_per_image, prompt_to_category, BASE_CATEGORIES)

    # 7. 显示结果
    print("\n" + "=" * 60)
    print("集成分类结果（使用{}个模板）".format(len(TEMPLATES_ZH)))
    print("=" * 60)

    predicted_idx = final_probs.argmax(dim=1).item()
    confidence = final_probs[0][predicted_idx].item() * 100

    print(f"\n🎯 预测类别：{BASE_CATEGORIES[predicted_idx]}")
    print(f"📊 置信度：{confidence:.2f}%")

    print("\n所有类别得分（按置信度排序）:")
    sorted_probs, sorted_indices = final_probs.sort(descending=True)
    for i, (prob, idx) in enumerate(zip(sorted_probs[0], sorted_indices[0]), 1):
        category = BASE_CATEGORIES[idx.item()]
        score = prob.item() * 100
        bar_length = int(score / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {i}. {category:8s} {score:6.2f}% {bar}")

    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
