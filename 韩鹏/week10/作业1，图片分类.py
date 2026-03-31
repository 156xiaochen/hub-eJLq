from PIL import Image
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# 官方 openai clip 不支持中文
# https://www.modelscope.cn/models/AI-ModelScope/chinese-clip-vit-base-patch16
model = ChineseCLIPModel.from_pretrained("./model/chinese-clip-vit-base-patch16")  # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("./model/chinese-clip-vit-base-patch16")  # 预处理

# ===================== 设备配置 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ===================== 1. 加载本地小狗图片 =====================
image_path = "./p1.jpg"
image = Image.open(image_path).convert("RGB")  # 解决透明图报错

# ===================== 2. 中文分类标签（零样本核心） =====================
# 想分什么类就写什么，纯中文！
text_labels = [
    "一只小狗",
    "一只小猫",
    "一辆汽车",
    "一个人",
    "一棵树",
    "一只兔子"
]

# ===================== 3. 图文预处理 =====================
inputs = processor(
    text=text_labels,
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

# ===================== 4. 模型推理 =====================
with torch.no_grad():
    outputs = model(**inputs)

# ===================== 5. 计算相似度 → 转概率 =====================
# ChineseCLIP 输出的是 图片-文本 相似度
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)  # 转为0~1概率

# ===================== 6. 获取分类结果 =====================
best_index = torch.argmax(probs, dim=1).item()
best_label = text_labels[best_index]
best_prob = probs[0][best_index].item() * 100

# ===================== 输出结果 =====================
print("="*50)
print(f"图片分类结果：{best_label}")
print(f"置信度：{best_prob:.2f}%")
print("="*50)


# ===================== 控制台打印 ===================
# 图片分类结果：一只小狗
# 置信度：93.69%
# ==================================================