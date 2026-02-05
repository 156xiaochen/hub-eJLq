# train.py
from modelscope import MsDataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import json

# 1. 加载数据
print("Loading dataset...")
data = list(MsDataset.load('winwin_inc/product-classification-hiring-demo', split='train'))[:500]
texts = [d['product_name'] for d in data]
labels_str = [d['category'] for d in data]

# 2. 标签编码
lbl = LabelEncoder()
labels = lbl.fit_transform(labels_str)
num_labels = len(lbl.classes_)

# 3. 划分训练/测试集
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# 4. Tokenizer & Model
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese", num_labels=num_labels
)

# 5. 编码
train_enc = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_enc = tokenizer(x_test, truncation=True, padding=True, max_length=64)

train_dataset = Dataset.from_dict({
    'input_ids': train_enc['input_ids'],
    'attention_mask': train_enc['attention_mask'],
    'labels': y_train
})
test_dataset = Dataset.from_dict({ 
    'input_ids': test_enc['input_ids'],
    'attention_mask': test_enc['attention_mask'],
    'labels': y_test
})

# 6. 计算指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}


training_args = TrainingArguments(
    output_dir='../assets/weights/bert/', # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,                  # 训练的总轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=100,                   # 每隔100步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
)

# 8. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset, 
    compute_metrics=compute_metrics,  
)

trainer.train()

# 9. 保存模型 + 类别映射
torch.save(model.state_dict(), "../assets/weights/bert.pt")
with open("../assets/weights/category_names.json", "w", encoding="utf-8") as f:
    json.dump(lbl.classes_.tolist(), f, ensure_ascii=False)

print("✅ 微调完成！模型已保存到 ../weights/bert/")
