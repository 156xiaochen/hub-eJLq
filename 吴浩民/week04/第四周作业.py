import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  # 修复了导入路径

# -------------------------- 1. 数据准备 --------------------------
# 使用你之前的路径逻辑
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)

# 标签处理
lbl = LabelEncoder()
lbl.fit(dataset[1].values)
num_labels = len(lbl.classes_)

# 划分数据
x_train, x_test, train_label, test_label = train_test_split(
    list(dataset[0].values[:500]),
    lbl.transform(dataset[1].values[:500]),
    test_size=0.2,
    stratify=dataset[1][:500].values
)

# 加载本地分词器
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')

# 编码
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# -------------------------- 2. 数据集和数据加载器 --------------------------
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encoding, train_label)
test_dataset = NewsDataset(test_encoding, test_label)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# -------------------------- 3. 模型和优化器 --------------------------
# 加载本地模型
model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese', num_labels=num_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optim = AdamW(model.parameters(), lr=2e-5)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# -------------------------- 4. 训练和验证函数 --------------------------
def train(epoch):
    model.train()
    total_train_loss = 0
    iter_num = 0
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        iter_num += 1
        if iter_num % 10 == 0: # 调小了打印频率，因为数据只有500条
            print(f"Epoch: {epoch}, iter_num: {iter_num}, loss: {loss.item():.4f}")

    print(f"Epoch: {epoch}, Average training loss: {total_train_loss / len(train_loader):.4f}")

def validation():
    model.eval()
    total_eval_accuracy = 0
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs[1].detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    print(f"Accuracy: {total_eval_accuracy / len(test_dataloader):.4f}")

# -------------------------- 5. 主训练循环 --------------------------
for epoch in range(4):
    print(f"------------Epoch: {epoch} ----------------")
    train(epoch)
    validation()

# -------------------------- 6. 新样本测试 (补全部分) --------------------------
def predict(text):
    model.eval()
    # 编码测试文本
    inputs = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    # 获取索引并转回中文标签
    pred_idx = torch.argmax(logits, dim=1).item()
    return lbl.inverse_transform([pred_idx])[0]

print("\n--- 最终测试验证 ---")
test_text = "中国队在本次比赛中表现出色，成功晋级决赛" # 这是一个新的输入
print(f"输入文本: {test_text}")
print(f"分类结果: 【{predict(test_text)}】")
