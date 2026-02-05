import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# --- 1. 数据预处理 ---
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40


class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# --- 2. 通用模型定义 ---
class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, model_type="LSTM"):
        super(SequenceClassifier, self).__init__()
        self.model_type = model_type.upper()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.model_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif self.model_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:  # LSTM
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        if self.model_type == "LSTM":
            _, (hn, cn) = self.rnn(embedded)
        else:
            _, hn = self.rnn(embedded)
        # 取最后一个隐藏层状态 (num_layers, batch, hidden) -> (batch, hidden)
        return self.fc(hn.squeeze(0))


# --- 3. 训练与精度对比函数 ---
def train_and_evaluate(model_type, epochs=5):
    print(f"\n--- 正在实验模型: {model_type} ---")

    train_loader = DataLoader(CharDataset(texts, numerical_labels, char_to_index, max_len), batch_size=32, shuffle=True)
    model = SequenceClassifier(vocab_size, 64, 128, len(label_to_index), model_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算精度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], 准确率: {accuracy:.2f}%")

    end_time = time.time()
    print(f"训练耗时: {end_time - start_time:.2f}s")
    return accuracy


# --- 4. 运行对比 ---
results = {}
for m in ["RNN", "LSTM", "GRU"]:
    acc = train_and_evaluate(m)
    results[m] = acc

# --- 5. 打印最终结果对比表 ---
print("\n" + "=" * 30)
print(f"{'模型类型':<10} | {'最终精度 (%)':<10}")
print("-" * 30)
for m, acc in results.items():
    print(f"{m:<10} | {acc:<10.2f}")
print("=" * 30)
