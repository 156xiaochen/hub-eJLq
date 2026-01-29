#1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/dataset.csv", encoding="utf-8", sep="\t", header=None)
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


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden1,hidden2,hidden3, output_dim):
        super(SimpleClassifier, self).__init__()
        #使用Sequential 来包装多层线性层，
        #引入relu 增加分线下表达
        #虽然每一个全连接层都叫线性层其实中间层数值越大，可训练的参数越多，最终输出的结果就会越复杂
        #就越能拟合各种函数
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_dim)
        )

    def forward(self, x):
        #简单的使用包装层即可，将数据塞入包装层，最后得到输出
        return self.layers(x)



char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, 128,128,128, output_dim) # 维度和精度有什么关系？
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

num_epochs = 100
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #if idx % 50 == 0:
        #    print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")


    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}


def test(text :str) ->str:
    predicted_class = classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label)
    result = f"{text} -> {predicted_class}"
    print(result)
    return result
#主要类别有:{'HomeAppliance-Control', 'Other', 'TVProgram-Play', 'Video-Play', 'Radio-Listen', 'Travel-Query', 'Weather-Query', 'Alarm-Update', 'Calendar-Query', 'FilmTele-Play', 'Music-Play', 'Audio-Play'}

#最后的结论，隐藏层设置的越多，单层的维度越多（可训练参数越多）训练时间越长，训练效果越好，loss最后越低
test("帮我导航到北京")
test("查询明天北京的天气")
test("我要怎么才能学好ai编程")
test("我想看和平精英上战神必备技巧的游戏视频")
test("播放钢琴曲命运交响曲")
test("我怎么去大梅沙")
test("小爱同学，汪汪队")
test("我要听广播1台")
