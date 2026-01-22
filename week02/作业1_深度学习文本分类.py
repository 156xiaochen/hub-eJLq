import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

#1. 数据预处理
dataset = pd.read_csv('../dataset.csv', sep='\t', header=None)
texts = dataset[0].tolist()
targets = dataset[1].tolist()

label_to_index = {label: idx for idx,label in enumerate(set(targets))}
# for label in targets:
#     if label not in label_to_index:
#         label_to_index[label] = len(label_to_index)
numerical_labels = [label_to_index[label] for label in targets]

char_to_index = {"<pad>": 0}
for text in texts:
    if text not in char_to_index:
        char_to_index[text] = len(char_to_index)
vocab_size = len(char_to_index)
max_len = 40

#2. dataset定义
class CharBoWDataset(Dataset):
    def __init__(self, texts, targets, char_to_index, vocab_size, max_len):
        self.texts = texts
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.char_to_index = char_to_index
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        all_bow_vectors = []
        for text in self.texts:
            tokenized = []
            for char in text[:self.max_len]:
                tokenized.append(self.char_to_index.get(char,0))

            while len(tokenized)<self.max_len:
                tokenized.append(0)
            bow_vector = torch.zeros(self.vocab_size)
            for idx in tokenized:
                if idx != 0:
                    bow_vector[idx] += 1
            all_bow_vectors.append(bow_vector)

        return torch.stack(all_bow_vectors)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.targets[idx]

    def __len__(self):
        return len(self.texts)

#3. 动态模型构建函数
def create_model(num_layers, hidden_dim, input_size, output_size):
    layers = [nn.Linear(input_size, hidden_dim), nn.ReLU()]

    for _ in range(num_layers-1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(hidden_dim, output_size))
    return nn.Sequential(*layers)

char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, vocab_size, max_len)
dataloader = DataLoader(dataset=char_dataset, batch_size=64, shuffle=True)

results = {}
#4. 训练函数定义
def train_and_record(model, model_name, num_epochs=10):
    print(f"\n开始训练模型:{model_name}")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()

    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch{epoch+1}/{num_epochs}",leave=True):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    results[model_name] = loss_history
    print(f"模型{model_name}训练完成")

#5. 执行对比实验
configurations = [
    (1, 64, "浅层网络，64节点"),
    (2, 128, "中层网络，128节点"),
    (3, 256, "深层网络，256节点")
]

for num_layers, hidden_dim, model_name in configurations:
    model = create_model(num_layers, hidden_dim, vocab_size, len(label_to_index))
    train_and_record(model, model_name)

#6. 绘图可视化
plt.figure(figsize=(12, 8))
for idx, (model_name, loss_list) in enumerate(results.items()):
    plt.plot(loss_list, label=model_name, linestyle='-', marker='o', linewidth=2)

plt.title('Compare',fontsize=16)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
