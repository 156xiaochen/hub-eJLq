"""
对比RNN、LSTM和GRU在文本分类任务上的性能
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# 加载数据
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# max length 最大输入的文本长度
max_len = 40

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# RNN模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state[-1])
        return out


# LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state[-1])
        return out


# GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state[-1])
        return out


def train_model(model, dataloader, num_epochs=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"  Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    return training_time


def evaluate_model(model, test_texts, test_labels, char_to_index, max_len, index_to_label):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for text, true_label_idx in zip(test_texts, test_labels):
            indices = [char_to_index.get(char, 0) for char in text[:max_len]]
            indices += [0] * (max_len - len(indices))
            input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
            
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_idx = predicted_idx.item()
            
            if predicted_idx == true_label_idx:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


# 准备数据
dataset_obj = TextDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset_obj, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

index_to_label = {i: label for label, i in label_to_index.items()}

print("="*60)
print("开始对比RNN、LSTM和GRU在文本分类任务上的性能")
print("="*60)

# 测试RNN
print("\n1. 训练RNN模型...")
rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
rnn_training_time = train_model(rnn_model, dataloader)
rnn_accuracy = evaluate_model(rnn_model, texts[:100], numerical_labels[:100], char_to_index, max_len, index_to_label)
print(f"  RNN训练时间: {rnn_training_time:.2f}s, 准确率(前100个样本): {rnn_accuracy:.4f}")

# 测试LSTM
print("\n2. 训练LSTM模型...")
lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_training_time = train_model(lstm_model, dataloader)
lstm_accuracy = evaluate_model(lstm_model, texts[:100], numerical_labels[:100], char_to_index, max_len, index_to_label)
print(f"  LSTM训练时间: {lstm_training_time:.2f}s, 准确率(前100个样本): {lstm_accuracy:.4f}")

# 测试GRU
print("\n3. 训练GRU模型...")
gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
gru_training_time = train_model(gru_model, dataloader)
gru_accuracy = evaluate_model(gru_model, texts[:100], numerical_labels[:100], char_to_index, max_len, index_to_label)
print(f"  GRU训练时间: {gru_training_time:.2f}s, 准确率(前100个样本): {gru_accuracy:.4f}")

print("\n" + "="*60)
print("性能对比总结:")
print("="*60)
print(f"RNN  - 训练时间: {rnn_training_time:.2f}s, 准确率: {rnn_accuracy:.4f}")
print(f"LSTM - 训练时间: {lstm_training_time:.2f}s, 准确率: {lstm_accuracy:.4f}")
print(f"GRU  - 训练时间: {gru_training_time:.2f}s, 准确率: {gru_accuracy:.4f}")
print("="*60)

# 测试预测效果
test_sentences = ["帮我导航到北京", "查询明天北京的天气", "播放音乐"]
print("\n预测结果对比:")

for sentence in test_sentences:
    print(f"\n输入句子: '{sentence}'")
    
    # RNN预测
    indices = [char_to_index.get(char, 0) for char in sentence[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    rnn_model.eval()
    lstm_model.eval()
    gru_model.eval()
    
    with torch.no_grad():
        rnn_pred = rnn_model(input_tensor)
        lstm_pred = lstm_model(input_tensor)
        gru_pred = gru_model(input_tensor)
        
        _, rnn_predicted = torch.max(rnn_pred, 1)
        _, lstm_predicted = torch.max(lstm_pred, 1)
        _, gru_predicted = torch.max(gru_pred, 1)
        
        print(f"  RNN 预测: {index_to_label[rnn_predicted.item()]}")
        print(f"  LSTM预测: {index_to_label[lstm_predicted.item()]}")
        print(f"  GRU 预测: {index_to_label[gru_predicted.item()]}")
