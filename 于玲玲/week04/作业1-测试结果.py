import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification

# BertForSequenceClassification bert 用于 文本分类
# Trainer： 直接实现 正向传播、损失计算、参数更新
# TrainingArguments： 超参数、实验设置
tokenizer = BertTokenizer.from_pretrained('./my_finetuned_bert')
model = BertForSequenceClassification.from_pretrained('./my_finetuned_bert')
model.eval()
# 单条测试
# new_text = "为什么结果都是一样的"
texts = [
    "你想不想去吃午饭？",
    "马上放假了，心情有点小激动呢！",
    "哦！我被选中了！",
    "真无语了，怎么会有这种人",
    "他真的有够恶心，卫生习惯有够差。"
]
inputs = tokenizer(
    texts,
    # new_text,
    padding=True,
    truncation=True,
    max_length=128,          # 必须与训练时一致！
    return_tensors="pt"      # 返回 PyTorch 张量
)
# print(type(model))
# 推理
# 禁用梯度计算（节省内存，加速）
with torch.no_grad():
    logits = model(**inputs).logits  # shape: [batch_size, num_labels]
    # print(logits)
# 后处理
predicted_ids = torch.argmax(logits, dim=-1).tolist()
# print(predicted_ids)

label_map = {}
# 加载和预处理数据
dataset_df = pd.read_csv("Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv", sep=",",skiprows=1, header=None)
# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 拟合数据并转换前500个标签，得到数字标签
labels = lbl.fit_transform(dataset_df[1])
print(labels)
for i, label in zip(labels,dataset_df[1]):
    label_map[i] = label
print(label_map)
print(predicted_ids)
for text, pred_id in zip(texts, predicted_ids):
# pred_id = predicted_ids[0]
#     print(text)
#     print(pred_id)
    prob = torch.softmax(logits[predicted_ids.index(pred_id)], dim=-1)[pred_id].item()
    print(f"「{text}」 → {label_map[pred_id]} (置信度: {prob:.2%})")
# print(logits)
# prob = torch.softmax(logits[0], dim=-1)[pred_id].item()
# print(f"「{new_text}」 → {label_map[pred_id]} (置信度: {prob:.2%})")