import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# --- 1. 设备配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"当前使用设备: {device}")

# --- 2. 数据预处理 ---
# 请确保路径正确，建议使用 r"" 防止转义
file_path = r"D:\Code\Python\badou\Week01\dataset.csv"
dataset_df = pd.read_csv(file_path, sep="\t", header=None)
texts = dataset_df[0].tolist()
string_labels = dataset_df[1].tolist()

# 标签映射
label_to_index = {label: i for i, label in enumerate(sorted(set(string_labels)))}
index_to_label = {i: label for label, i in label_to_index.items()}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建词表 (增加 <unk> 处理未知字符)
char_to_index = {'<pad>': 0, '<unk>': 1}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)
max_len = 40

# --- 3. 数据集定义 ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.data = []
        for text, label in zip(texts, labels):
            # 将文本转为索引，不在词表中的转为 <unk> (索引1)
            indices = [char_to_index.get(char, 1) for char in text[:max_len]]
            indices += [0] * (max_len - len(indices))
            self.data.append((torch.tensor(indices), torch.tensor(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- 4. 模型定义 ---
class BaseClassifier(nn.Module):
    def __init__(self, mode, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if mode == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif mode == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif mode == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        # 对于不同模型统一处理隐藏状态
        out, h_state = self.rnn(x)
        
        # 如果是 LSTM, h_state 是 (h, c) 元组，取 h
        if isinstance(h_state, tuple):
            h_state = h_state[0]
            
        # 取最后一层的最后一个时间步的隐藏状态
        return self.fc(h_state[-1])

# --- 5. 训练与评估函数 ---
def run_experiment(model_type, train_loader, test_loader, vocab_size, emb_dim, hid_dim, out_dim):
    print(f"\n--- 正在训练 {model_type} 模型 ---")
    model = BaseClassifier(model_type, vocab_size, emb_dim, hid_dim, out_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(train_loader):.4f}")

    # 评估
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"{model_type} 准确率: {accuracy:.2f}%")
    return model

# --- 6. 执行流水线 ---
full_dataset = TextDataset(texts, numerical_labels, char_to_index, max_len)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 参数配置
emb_dim, hid_dim = 64, 128
out_dim = len(label_to_index)

# 训练三个模型
rnn_model = run_experiment("RNN", train_loader, test_loader, vocab_size, emb_dim, hid_dim, out_dim)
lstm_model = run_experiment("LSTM", train_loader, test_loader, vocab_size, emb_dim, hid_dim, out_dim)
gru_model = run_experiment("GRU", train_loader, test_loader, vocab_size, emb_dim, hid_dim, out_dim)

# --- 7. 预测示例 ---
def predict(text, model):
    model.eval()
    indices = [char_to_index.get(char, 1) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.max(output, 1)[1].item()
    return index_to_label[pred_idx]

print("\n--- 测试预测 ---")
test_samples = ["帮我导航到北京", "查询明天北京的天气", "播放音乐"]
for ts in test_samples:
    print(f"输入: {ts} | RNN: {predict(ts, rnn_model)} | LSTM: {predict(ts, lstm_model)} | GRU: {predict(ts, gru_model)}")