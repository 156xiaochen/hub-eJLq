import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ... (数据加载和预处理保持不变) ...
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

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


# 可配置的模型类
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FlexibleClassifier, self).__init__()
        layers = []
        last_dim = input_dim

        # 动态构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 准备数据
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
output_dim = len(label_to_index)
num_epochs = 10

# 测试不同的模型配置
configs = [
    ([64], "1-layer-64-node"),
    ([128], "1-layer-128-node"),
    ([256], "1-layer-256-node"),
    ([128, 64], "2-layer-128-64-node"),
    ([256, 128], "2-layer-256-128-node"),
]

# 存储每个配置的loss历史
loss_history = {}

print("开始对比不同模型结构的Loss变化...")

for hidden_dims, config_name in configs:
    print(f"\n=== 训练配置: {config_name} ===")

    model = FlexibleClassifier(vocab_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    config_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        config_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    loss_history[config_name] = config_losses

# 可视化对比结果
plt.figure(figsize=(10, 6))
for config_name, losses in loss_history.items():
    plt.plot(range(1, num_epochs + 1), losses, label=config_name, marker='o')

plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('损失值 (Loss)')
plt.title('不同模型结构的Loss变化对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 打印最终结果对比
print("\n" + "=" * 50)
print("各模型最终Loss对比")
print("=" * 50)

# 按最终loss排序
sorted_results = sorted(loss_history.items(), key=lambda x: x[1][-1])

for config_name, losses in sorted_results:
    final_loss = losses[-1]
    print(f"{config_name}: {final_loss:.4f}")

best_config = sorted_results[0]
print(f"\n最佳模型: {best_config[0]}, 最终Loss: {best_config[1][-1]:.4f}")
