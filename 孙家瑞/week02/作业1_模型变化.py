from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 加载数据
dataset = pd.read_csv("../../week01/code/dataset.csv", sep="\t", header=None)
# 输入的文本
texts = dataset[0].tolist()
# 分类标签
string_labels = dataset[1].tolist()

# 构建一个dict,key是分类标签，value是一个索引id
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将所有标签转换为数字索引
numerical_labels = [label_to_index[label] for label in string_labels]
# 构建一个文本字典，字符->索引
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
# 数字->字符
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)  # 字符集大小，共有2823个字符
# 每个输入的最大长度
max_len = 40


# 构建一个自定义数据集
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

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


# 自定义模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layer, output_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential()
        last_dim = input_dim
        # 根据输入的数组，构建一个神经网络
        for n_dim in hidden_layer:
            self.network.append(nn.Linear(last_dim, n_dim))
            self.network.append(nn.ReLU())
            last_dim = n_dim
        # 最后设置一个输出层
        self.network.append(nn.Linear(last_dim, output_dim))
        # 这里直接定义损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y_true=None):
        # 手动实现每层的计算
        y_pred = self.network(x)
        # 如果有y的真实值，输出是损失函数的计算结果
        # 没有的时候，就输出预测值
        if y_true is not None:
            return self.loss(y_pred, y_true)
        else:
            return y_pred


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)  # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)  # 读取批量数据集 -》 batch数据

output_dim = len(label_to_index)


def train_model(hiddens: List[int], num_epochs=10, learning_rate=0.01):
    """
    训练模型，并返回损失函数的变化
    :param hiddens: 模型隐藏层的形状
    :param num_epochs: 迭代次数
    :param learning_rate: 学习率
    :return: 损失函数的变化情况，数组
    """
    print(
        f"==============================start training{'-'.join(str(layer) for layer in hiddens)}============================")
    # epoch： 将数据集整体迭代训练一次
    epochs = num_epochs
    # 学习率
    lr = learning_rate

    model = SimpleClassifier(vocab_size, hiddens, output_dim)  # 维度和精度有什么关系？
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # 用于记录 损失函数的变化情况
    watching_loss = []

    # batch： 数据集汇总为一批训练一次
    for epoch in range(epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()  # 梯度归零
            loss = model(inputs, labels)  # 因为有预测值，这里会计算损失函数
            loss.backward()  # 计算梯度
            optimizer.step()  # 使用优化器调整参数
            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
        watching_loss.append(running_loss / len(dataloader))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    return watching_loss


# train_model(hidden_dim)

# 绘制图像
def plot_graph(model_infos) -> None:
    plt.figure(figsize=(10, 6))
    for idx, (loss, hidden, m_color) in enumerate(model_infos):
        plt.plot(range(len(loss)), loss, label=f'Model: {"-".join([str(h) for h in hidden])}', color=m_color,
                 linewidth=1)
    plt.xlabel('epoch_cnt')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    hidden_layers = [[64], [128], [256, 128], [64, 256, 32], [10, 100, 200, 300]]
    color = ['orange', 'red', 'green', 'blue', 'yellow']
    data = []
    for i, hidden_layer in enumerate(hidden_layers):
        data.append([train_model(hidden_layer), hidden_layer, color[i]])
    plot_graph(data)
