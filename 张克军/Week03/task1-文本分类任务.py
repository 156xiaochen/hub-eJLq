from abc import abstractmethod, ABC

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt

"""
1、 理解rnn、lstm、gru的计算过程（面试用途），阅读官方文档 ：https://docs.pytorch.org/docs/2.4/nn.html#recurrent-layers
 最终 使用 GRU 代替 LSTM 实现05_LSTM文本分类.py；05_LSTM文本分类.py 使用lstm ，使用rnn/ lstm / gru 分别代替原始lstm，进行实验，对比精度
"""

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

print(f"label分类的总数:{len(label_to_index)} ,总文章数量:{len(numerical_labels)}")
#当前任务是文本分类，当前采用字符向量的方式，即输入字符列表进行训练分类 ，不用分词，所以拿到字符的对应索引，后面这个索引就变成embading向量
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

print(f"字符总数量:{vocab_size}")

# max length 最大输入的文本长度
max_len = 40

# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharRnnDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应的标签
        self.char_to_index = char_to_index # 字符到索引的映射关系
        self.max_len = max_len # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        #拿到每个文章字符最大max_len的长度，不足的用0填充，将其转为索引输出，预训练结果（标签）是对应分类索引
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

#RNN家族类的 循环神经网络组合
class RnnBaseClassifier(nn.Module,ABC):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RnnBaseClassifier, self).__init__()
        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.fc = nn.Linear(hidden_dim, output_dim) #最终给个线性分类层 batch*hidden_dim -> batch*output_dim

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        hidden_data = self.rnn_2_hidden(embedded)
        # batch size * output_dim
        out = self.fc(hidden_data)
        return out

    @abstractmethod
    def rnn_2_hidden(self, embedded):
        """RNN层处理嵌入向量，子类必须实现此方法"""
        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        pass

class LSTMClassifier(RnnBaseClassifier):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层

    def rnn_2_hidden(self, embedded):
        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        return hidden_state.squeeze(0)
class GRUClassifier(RnnBaseClassifier):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def rnn_2_hidden(self, embedded):
        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        gru_out, hidden_state = self.gru(embedded)
        return hidden_state.squeeze(0)
class RnnClassifier(RnnBaseClassifier):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RnnClassifier, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

    def rnn_2_hidden(self, embedded):
        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        rnn_out, hidden_state = self.rnn(embedded)
        return hidden_state.squeeze(0)


def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# --- Training and Prediction ---
lstm_dataset = CharRnnDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model_dics = {
    "LSTM": LSTMClassifier,
    "GRU": GRUClassifier,
    "RNN": RnnClassifier
}
index_to_label = {i: label for label, i in label_to_index.items()}
num_epochs = 10

# 用于存储每个模型的时间和平均loss记录
model_records = {}

for key in model_dics:
    print(f"\n{'='*50}")
    print(f"开始训练 {key} 模型")
    print(f"{'='*50}")
    
    model = model_dics[key](vocab_size, embedding_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 初始化记录
    model_records[key] = {
        'model': model,
        'epoch_losses': [],  # 每个epoch的平均loss
        'total_time': 0      # 总训练时间
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if idx % 50 == 0:
                current_time = time.time() - start_time
                print(f"{key} Epoch {epoch+1}, Batch {idx}, Time: {current_time:.2f}s, Loss: {loss.item():.6f}")

        # 记录每个epoch的平均loss
        avg_loss = running_loss / len(dataloader)
        model_records[key]['epoch_losses'].append(avg_loss)
        
        epoch_time = time.time() - start_time
        print(f"{key} Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.6f}, 总时间: {epoch_time:.2f}s")

    # 记录总训练时间
    model_records[key]['total_time'] = time.time() - start_time

# 绘制简化对比图
plt.figure(figsize=(12, 5))
# 绘制平均Loss曲线对比
plt.subplot(1, 2, 1)
for model_name, records in model_records.items():
    plt.plot(range(1, len(records['epoch_losses']) + 1), records['epoch_losses'], 
             label=f'{model_name}', linewidth=2, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Loss vs Epoch Comparison')
plt.legend()
plt.grid(True)

# 绘制总训练时间对比
plt.subplot(1, 2, 2)
model_names = list(model_records.keys())
total_times = [records['total_time'] for records in model_records.values()]
colors = ['skyblue', 'lightcoral', 'lightgreen']
bars = plt.bar(model_names, total_times, color=colors)
plt.xlabel('Model Type')
plt.ylabel('Total Training Time (seconds)')
plt.title('Total Training Time Comparison')
for bar, time_val in zip(bars, total_times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{time_val:.2f}s', ha='center', va='bottom')
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('task1_res.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印简化总结
new_text = "帮我导航到北京"
new_text_2 = "查询明天北京的天气"

print("\n" + "="*50)
print(f"训练总结报告 预测文本1:'{new_text}' 预测文本2:'{new_text_2}'")
print("="*50)
for model_name, records in model_records.items():
    final_loss = records['epoch_losses'][-1]  # 最后一个epoch的loss
    total_time = records['total_time']
    model = records['model']

    predicted_class = classify_text(new_text, model, char_to_index, max_len, index_to_label)
    predicted_class_2 = classify_text(new_text_2, model, char_to_index, max_len, index_to_label)

    print(f"{model_name:8} | 总时间: {total_time:6.2f}s | 最终Loss: {final_loss:.6f} 结果1:'{predicted_class}' 结果2:'{predicted_class_2}'")
print("="*50)

"""
==================================================
训练总结报告 预测文本1:'帮我导航到北京' 预测文本2:'查询明天北京的天气'
==================================================
LSTM     | 总时间:  24.00s | 最终Loss: 0.142596 结果1:'Travel-Query' 结果2:'Weather-Query'
GRU      | 总时间:  44.50s | 最终Loss: 0.047741 结果1:'Travel-Query' 结果2:'Weather-Query'
RNN      | 总时间:  20.96s | 最终Loss: 2.331388 结果1:'FilmTele-Play' 结果2:'FilmTele-Play'

就这个结果而言，GRU效果最好 但是最耗时,rnn无法收敛 不确定为什么
还有学习率也是一个关键参数 ，当学习率设置0.01，LSTM和RNN都没有太大变化，GRU到后面loss越来越大，很是奇怪，不知原因
"""