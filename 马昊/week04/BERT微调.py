import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
import joblib
# -------------------------- 1. 数据加载与预处理（修复核心错误） --------------------------
# 修复：去掉sep="\t"（CSV默认逗号分隔），指定列名，读取完整数据
dataset_df = pd.read_csv(
    r"C:\Users\48628\Desktop\text_classification_1000_shuffled.csv",
    encoding="utf-8",  # 适配Windows默认编码
    header=None,      # 数据集无表头
    names=["text", "label"]  # 手动指定列名（文本列=0，标签列=1）
)

# 修复：使用完整数据集（去掉[:500]），避免数据量不足
# 初始化LabelEncoder，转换文本标签为数字
lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_df["label"].values)  # 修复变量名：label→labels
texts = dataset_df["text"].values.tolist()              # 完整文本数据

# 修复：变量名统一（texts/labels，避免拼写错误）
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,                # 正确变量名：text→texts
    labels,               # 正确变量名：label→labels
    test_size=0.2,
    stratify=labels,      # 分层抽样，保证标签分布一致
    random_state=42       # 固定种子，结果可复现
)

# -------------------------- 2. 加载模型和分词器（修复路径） --------------------------
# 修复：使用Hugging Face官方路径（自动下载bert-base-chinese）
# 如果本地有模型，替换为本地绝对路径（如r"C:\models\bert-base-chinese"）
tokenizer = BertTokenizer.from_pretrained(r"C:\models\google-bert\bert-base-chinese")
# 修复：自动计算标签数量（避免硬编码8，更通用）
num_labels = len(lbl.classes_)
model = BertForSequenceClassification.from_pretrained(
    r"C:\models\google-bert\bert-base-chinese",
    num_labels=num_labels,
    ignore_mismatched_sizes=True  # 避免预训练模型权重维度不匹配
)

# -------------------------- 3. 文本编码 --------------------------
# 批量编码，添加return_tensors="pt"确保格式正确
train_encodings = tokenizer(
    x_train,
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors="pt"
)
test_encodings = tokenizer(
    x_test,
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors="pt"
)

# -------------------------- 4. 构建Dataset对象 --------------------------
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels.astype(np.int64)  # 确保标签是int64类型
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels.astype(np.int64)
})

# -------------------------- 5. 评估函数（修复变量名） --------------------------


def compute_metrics(eval_pred):
    logits, label_eval = eval_pred  # label_eval是真实标签
    predictions = np.argmax(logits, axis=-1)
    # 修复：用label_eval代替labels（labels是全局变量，这里要用当前批次的真实标签）
    accuracy = (predictions == label_eval).mean()
    return {'accuracy': accuracy}


# -------------------------- 6. 训练参数配置 --------------------------
training_args = TrainingArguments(
    output_dir=r"C:\Users\48628\Desktop\text_classification_model_results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=r'C:\Users\48628\Desktop\text_classification_model_results\logs',  # 绝对路径，避免报错
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # 按准确率选最优模型
    fp16=False,  # 关闭混合精度（CPU训练时必须关）
    disable_tqdm=False,  # 显示训练进度条
)

# -------------------------- 7. 训练模型 --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()
# 最终评估
eval_results = trainer.evaluate()
print("最终评估结果：", eval_results)

# 保存模型和标签编码器
model.save_pretrained(r"C:\Users\48628\Desktop\text_classification_model_results\final_model")
tokenizer.save_pretrained(r"C:\Users\48628\Desktop\text_classification_model_results\final_tokenizer")
# 保存LabelEncoder，方便后续预测
joblib.dump(lbl, r"C:\Users\48628\Desktop\text_classification_model_results\label_encoder.pkl")
