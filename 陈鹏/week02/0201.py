# 调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化

import pandas as pd
import torch  # 导入 PyTorch 核心库，用于构建和训练神经网络
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，包含层、激活函数、损失函数等
import torch.optim as optim  # 导入 PyTorch 的优化器模块，用于定义梯度下降等优化方法
from torch.utils.data import Dataset, DataLoader  # 导入数据集相关工具
import matplotlib.pyplot as plt # 绘图

# 数据加载预处理
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

max_len = 40

class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    # 生成词袋（BoW）向量的核心方法
    """
        遍历每个文本，截取前 40 个字符，转为字符索引（未知字符用 0 填充）。
        若文本长度不足 40，补充 0（<pad>）至 40 位，得到tokenized_texts（所有文本的字符索引列表）
    """
    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

# 模型1：原模型（2层，隐藏层128节点）
class Model1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 模型2：2层，隐藏层32节点（减少节点数）
class Model2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 节点数改为32
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 模型3：3层，隐藏层128→64节点（增加层数）
class Model3(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Model3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 新增隐藏层
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

def record_loss(model, dataloader, criterion , optimizer, num_epochs = 10):

    loss_history = []  # 记录每轮的平均Loss
    for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model.train()  # 将模型设为训练模式
        running_loss = 0.0  # 累计每个 epoch 的损失，用于计算平均损失
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()  # 清空上一轮的梯度（PyTorch 梯度会累加，必须手动清空）
            outputs = model(inputs)  # 前向传播，得到模型输出（每个样本的类别得分）
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 根据梯度更新模型参数（梯度下降）
            running_loss += loss.item()  # 累加损失（item()取出 Tensor 的数值，避免显存占用）
            if idx % 100 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return loss_history

# 初始化数据集
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
output_dim = len(label_to_index)

num_epochs = 10
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
lr = 0.01 # 学习率

# 训练模型1（原模型）
print("="*30 + " 训练原模型（2层，128节点） " + "="*30)
model1 = Model1(vocab_size, 128, output_dim)
optimizer1 = optim.SGD(model1.parameters(), lr=lr)
loss1 = record_loss(model1, dataloader, criterion, optimizer1, num_epochs)

# 训练模型2（2层，32节点）
print("\n" + "="*30 + " 训练模型2（2层，32节点） " + "="*30)
model2 = Model2(vocab_size, 32, output_dim)
optimizer2 = optim.SGD(model2.parameters(), lr=lr)
loss2 = record_loss(model2, dataloader, criterion, optimizer2, num_epochs)

# 训练模型3（3层，128→64节点）
print("\n" + "="*30 + " 训练模型3（3层，128→64节点） " + "="*30)
model3 = Model3(vocab_size, 128,64, output_dim)
optimizer3 = optim.SGD(model3.parameters(), lr=lr)
loss3 = record_loss(model3, dataloader, criterion, optimizer3, num_epochs)

# 打印每轮Loss对比
print("\n" + "="*50 + " Loss对比结果 " + "="*50)
print("Epoch\t原模型(2层128)\t模型2(2层32)\t模型3(3层128→64)")
for i in range(num_epochs):
    print(f"{i+1}\t{loss1[i]:.4f}\t\t{loss2[i]:.4f}\t\t{loss3[i]:.4f}")

# ========== 新增：解决Matplotlib中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文（Windows系统）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题
# ========== 字体配置结束 ==========


# 简单可视化
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs+1), loss1, label="原模型（2层128节点）")
plt.plot(range(1, num_epochs+1), loss2, label="模型2（2层32节点）")
plt.plot(range(1, num_epochs+1), loss3, label="模型3（3层128→64节点）")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("不同模型结构Loss变化对比")
plt.legend()
plt.grid(True)
plt.show()

# 预测模型
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    # 生成新文本的词袋向量（统计字符出现次数）
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    # 给向量增加一个维度（从[vocab_size]转为[1, vocab_size]），匹配模型输入的 batch 维度（模型默认接收批量数据）
    bow_vector = bow_vector.unsqueeze(0)

    model.eval() # 将模型设为评估模式
    with torch.no_grad(): # 禁用梯度计算
        output = model(bow_vector) # 前向传播得到预测得分

    _, predicted_index = torch.max(output, 1) # 在维度 1（类别维度）取最大值，返回（最大值，索引），索引即预测的类别编号
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# 构建数字索引到字符串标签的反向映射（用于预测结果转义）
index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到武汉"
predicted_class = classify_text(new_text, model1, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

