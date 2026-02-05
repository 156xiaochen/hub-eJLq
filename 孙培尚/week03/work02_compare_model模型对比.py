# compare_rnn_lstm_gru.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. 数据加载与预处理（统一一次）
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

# 划分训练集和测试集（关键！）
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels
)

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

# 2. 定义三种模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)  # hidden: (1, batch, hidden_dim)
        return self.fc(hidden.squeeze(0))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)  # hidden: (1, batch, hidden_dim)
        return self.fc(hidden.squeeze(0))


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)  # hidden: (1, batch, hidden_dim)
        return self.fc(hidden.squeeze(0))


# 3. 评估函数（在测试集上计算 accuracy）
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# 4. 训练并返回测试准确率
def train_and_evaluate(model_class, model_name, train_loader, test_loader, config, device):
    print(f"\nTraining {model_name}...")
    model = model_class(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['num_epochs']):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 在测试集上评估
    test_acc = evaluate_model(model, test_loader, device)
    print(f"{model_name} Test Accuracy: {test_acc:.2f}%")
    return test_acc



if __name__ == "__main__":
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 超参数配置（统一！）
    config = {
        'vocab_size': vocab_size,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'output_dim': len(label_to_index),
        'lr': 0.001,
        'num_epochs': 4,
        'batch_size': 32
    }

    # 创建 DataLoader
    train_dataset = CharDataset(train_texts, train_labels, char_to_index, max_len)
    test_dataset = CharDataset(test_texts, test_labels, char_to_index, max_len)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 定义要对比的模型
    models = {
        "RNN": RNNClassifier,
        "LSTM": LSTMClassifier,
        "GRU": GRUClassifier
    }

    # 运行对比
    results = {}
    for name, model_class in models.items():
        acc = train_and_evaluate(model_class, name, train_loader, test_loader, config, device)
        results[name] = acc

    # 打印最终对比结果
    print("\n" + "="*50)
    print("=== Model Comparison")
    for name, acc in results.items():
        print(f"{name:<6}: {acc:.2f}%")
    print("="*50)
