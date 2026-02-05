import pandas as pd  # 用于数据读取和处理
import torch  # PyTorch核心库，用于张量计算和深度学习
import torch.nn as nn  # PyTorch的神经网络模块
import torch.optim as optim  # 优化器模块
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器
import matplotlib.pyplot as plt  # 绘图

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()  # 提取第0列（文本）转为列表
string_labels = dataset[1].tolist()  # 提取第1列（标签）转为列表

# 将字符串标签（如 "导航"）映射为数字索引（如 0），set(string_labels)去重得到所有唯一标签，enumerate生成索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将所有字符串标签转为对应的数字索引，方便模型计算（模型只能处理数值）
numerical_labels = [label_to_index[label] for label in string_labels]

# 原始的文本构建一个词典，字 -》 数字
# 遍历 “帮我导航到北京” 后，
# char_to_index可能是{'<pad>':0, '帮':1, '我':2, '导':3, '航':4, ...}
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 生成 “数字→字符” 的反向映射字典（用于后续调试 / 还原）
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# max length 最大输入的文本长度
max_len = 40


# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharLSTMDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts  # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long)  # 文本对应的标签
        self.char_to_index = char_to_index  # 字符到索引的映射关系
        self.max_len = max_len  # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        # 根据索引获取单个样本（必须实现）
        text = self.texts[idx]
        # 1. 把文本转为字符编号，只取前max_len个字符
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        # 2. 长度不足max_len时，用<pad>（编号0）填充
        indices += [0] * (self.max_len - len(indices))
        # 3. 返回字符编号张量 + 标签（模型输入必须是张量）
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        # 1. 嵌入层：把字符编号（如1,2,3）转为稠密向量（如[0.1,0.2,...]）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        # 2. LSTM层：处理序列数据，batch_first=True表示输入格式为[batch_size, seq_len, embedding_dim]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        # 3. 全连接层：把LSTM的输出映射到标签数量（分类）
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 前向传播：定义数据如何通过模型
        embedded = self.embedding(x)
        # LSTM输出：lstm_out是所有时间步的输出，(h,c)是最后一个时间步的隐藏状态和细胞状态
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        # 统一使用最后一个时间步的输出（和GRU/RNN保持一致）
        out = self.fc(lstm_out[:, -1, :])
        return out


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        # 1. 嵌入层：把字符编号（如1,2,3）转为稠密向量（如[0.1,0.2,...]）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        # 2. GRU层：处理序列数据，batch_first=True表示输入格式为[batch_size, seq_len, embedding_dim]
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        # 3. 全连接层：把GRU的输出映射到标签数量（分类）
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 前向传播：定义数据如何通过模型
        embedded = self.embedding(x)
        # GRU输出
        gru_out, hidden_state = self.gru(embedded)
        # 用最后一个时间步的隐藏状态做分类（squeeze(0)去掉多余维度）
        out = self.fc(gru_out[:, -1, :])
        return out


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        # 1. 嵌入层：把字符编号（如1,2,3）转为稠密向量（如[0.1,0.2,...]）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        # 2. RNN层：处理序列数据，batch_first=True表示输入格式为[batch_size, seq_len, embedding_dim]
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        # 3. 全连接层：把rnn的输出映射到标签数量（分类）
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 前向传播：定义数据如何通过模型
        embedded = self.embedding(x)
        # RNN输出
        rnn_out, hidden_state = self.rnn(embedded)
        # 用最后一个时间步的隐藏状态做分类（squeeze(0)去掉多余维度）
        # 在 PyTorch/Numpy 的张量索引中，: 表示 “取这个维度的所有元素”，-1 表示 “取这个维度的最后一个元素”
        out = self.fc(rnn_out[:, -1, :])
        return out


def record_loss(model, model_name, dataloader, criterion, optimizer, num_epochs=4):
    loss_history = []  # 记录每轮的平均Loss
    for epoch in range(num_epochs):
        model.train()  # 模型设为训练模式（启用Dropout/BatchNorm等）
        running_loss = 0.0  # 累计损失
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # 清空上一轮的梯度（必须）
            outputs = model(inputs)  # 前向传播：输入→模型→输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播：计算梯度
            optimizer.step()  # 更新参数
            running_loss += loss.item()  # 累加损失值
            if idx % 100 == 0:
                print(f"{model_name} - Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"{model_name} - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return loss_history


# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
# 批量加载数据，shuffle=True 表示训练时打乱数据（提升泛化能力）
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
# 选设备：有GPU用GPU，没有用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 4
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（分类任务常用）

model1 = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)  # Adam优化器，学习率0.001
loss1 = record_loss(model1, "LSTM", dataloader, criterion, optimizer1, num_epochs)

model2 = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)  # Adam优化器，学习率0.001
loss2 = record_loss(model2, "GRU", dataloader, criterion, optimizer2, num_epochs)

model3 = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)  # Adam优化器，学习率0.001
loss3 = record_loss(model3, "RNN", dataloader, criterion, optimizer3, num_epochs)

# 打印每轮Loss对比
print("\n" + "=" * 50 + " Loss对比结果 " + "=" * 50)
for i in range(num_epochs):
    print(f"{i + 1}\t{loss1[i]:.4f}\t\t{loss2[i]:.4f}\t\t{loss3[i]:.4f}")

# ========== 新增：解决Matplotlib中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文（Windows系统）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# ========== 字体配置结束 ==========


# 简单可视化
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), loss1, label="LSTM")
plt.plot(range(1, num_epochs + 1), loss2, label="GRU")
plt.plot(range(1, num_epochs + 1), loss3, label="RNN")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("不同模型结构Loss变化对比")
plt.legend()
plt.grid(True)
plt.show()


def classify_text_lstm(text, model, char_to_index, max_len, index_to_label, device):
    # 1. 文本→字符编号，统一长度（和训练时的预处理一致）
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    # 增加batch维度（模型输入需要batch_size，这里是1）
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()  # 模型设为评估模式（禁用Dropout/BatchNorm）
    with torch.no_grad():  # 禁用梯度计算
        output = model(input_tensor)  # 前向传播得到预测得分

    _, predicted_index = torch.max(output, 1)  # 取得分最大的标签索引
    predicted_index = predicted_index.item()  # 张量→普通数字
    predicted_label = index_to_label[predicted_index]  # 数字→原始标签

    return predicted_label


# 数字→标签的反向映射
index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model1, char_to_index, max_len, index_to_label, device)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model1, char_to_index, max_len, index_to_label, device)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
