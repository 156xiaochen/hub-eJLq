import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# BertForSequenceClassification bert 用于 文本分类
# Trainer： 直接实现 正向传播、损失计算、参数更新
# TrainingArguments： 超参数、实验设置

from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 加载和预处理数据
dataset_df = pd.read_csv("Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv", sep=",",skiprows=1, header=None)

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 拟合数据并转换前500个标签，得到数字标签
labels = lbl.fit_transform(dataset_df[1])
# 提取前500个文本内容
texts = list(dataset_df[0])

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels    # 确保训练集和测试集的标签分布一致
)



model_path = "/Users/yull/AI/model/bert-base-chinese"
# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=8)

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=64：最大序列长度
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=128)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码
    'labels': train_labels                               # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}

# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,                  # 训练的总轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练， step 定义为 一次 正向传播 + 参数更新
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=200,                   # 每隔100步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 实例化 Trainer 简化模型训练代码
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

# 深度学习训练过程，数据获取，epoch batch 循环，梯度计算 + 参数更新

# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()
trainer.save_model("./my_finetuned_bert")
tokenizer.save_pretrained("./my_finetuned_bert")
# trainer 是比较简单，适合训练过程比较规范化的模型
# 如果我要定制化训练过程，trainer无法满足
# trainer.model.eval()
print(trainer.state.best_model_checkpoint)
# trainer.model.eval()
# # 单条测试
# # new_text = "放一首周杰伦的歌"
# texts = [
#     "这个电影真的挺好看的！",
#     "马上放假了，心情有点小激动呢！",
#     "真无语了，怎么会有这种人"
# ]
# inputs = tokenizer(
#     texts,
#     padding=True,
#     truncation=True,
#     max_length=64,          # 必须与训练时一致！
#     return_tensors="pt"      # 返回 PyTorch 张量
# )
# # print(type(model))
# # 推理
# # 禁用梯度计算（节省内存，加速）
# with torch.no_grad():
#     logits = trainer.model(**inputs).logits  # shape: [batch_size, num_labels]
#     # print(logits)
# # 后处理
# predicted_ids = torch.argmax(logits, dim=-1).tolist()
# # print(predicted_ids)
#
# label_map = {}
# # 加载和预处理数据
# dataset_df = pd.read_csv("Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv", sep=",",skiprows=1, header=None)
# # 初始化 LabelEncoder，用于将文本标签转换为数字标签
# lbl = LabelEncoder()
# # 拟合数据并转换前500个标签，得到数字标签
# labels = lbl.fit_transform(dataset_df[1])
# print(labels)
# for i, label in zip(labels,dataset_df[1]):
#     label_map[i] = label
# print(label_map)
# print(predicted_ids)
# for text, pred_id in zip(texts, predicted_ids):
# # pred_id = predicted_ids[0]
# #     print(text)
# #     print(pred_id)
#     prob = torch.softmax(logits[predicted_ids.index(pred_id)], dim=-1)[pred_id].item()
# # print(logits)
# # prob = torch.softmax(logits[0], dim=-1)[pred_id].item()
#     print(f"「{text}」 → {label_map[pred_id]} (置信度: {prob:.2%})")