#2、调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。

import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

#定义一个模拟数据数量
Numbers = 5000

# 1. 生成模拟数据 ，模拟数据 范围 大一点，放大到 0 - 32 ,这样 sin周期多一点
x_numpy = np.random.rand(Numbers, 1) * 10 # >3π
# 形状为 (Numbers, 1) 的二维数组，其中包含 Numbers 个在 [0, 32) 范围内均匀分布的随机浮点数。
#后面的噪声，设置得小一点 ，让sin函数 看起来更真实一点
y_numpy = np.sin(x_numpy) * 10 + np.random.randn(Numbers, 1) * 0.2
           #+ np.random.randn(Numbers, 1)*0.5)

#要拟合的是sin函数 且要求用多层网络结构你和 ，建立一个多层结果类
#这里使用一个简单的多层网络结构，暂时定义两个中间层，总共4层网络，因为当前任务足够简单
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden1,hidden2,hidden3, output_dim):
        super(SimpleClassifier, self).__init__()
        #使用Sequential 来包装多层线性层，
        #中间引入Tanh，让数据增加非线性表达，比relu更好，relu的非线性效果不好
        #虽然每一个全连接层都叫线性层其实中间层数值越大，可训练的参数越多，最终输出的结果就会越复杂
        #就越能拟合各种函数
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, hidden3),
            nn.Tanh(),
            nn.Linear(hidden3, output_dim)
        )

    def forward(self, x):
        #简单的使用包装层即可，将数据塞入包装层，最后得到输出
        return self.layers(x)


X = torch.from_numpy(x_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

#定义一个数据集，直接取自两个tensor数据集
dataset = TensorDataset(X, y)
#定义一个数据加载器，，后续直接可以用，方便数据打乱之类的操作
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)


print("数据生成完成。")
print("---" * 10)
# 2. 定义模型
model = SimpleClassifier(input_dim=1, hidden1=128, hidden2=128, hidden3=128, output_dim=1)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
#优化器的概念，其实本质就是对可训练参数，求梯度（偏导）（导数和），然后对这个导数计算结果 来更新可训练参数
# 所谓导数即数据变化方向，拿到损失函数的变化变化，自动王损失更低的方向调参，
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # lr 是学习率

# 4. 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    epoch_total_loss = 0.0  # 记录当前epoch的总损失
    num_batches = 0  # 记录批次数量
    #直接从数据加载器中获取批量数据
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 累积损失值
        epoch_total_loss += loss.item()
        num_batches += 1
    avg_loss = epoch_total_loss / num_batches
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.6f}')

# 5. 打印最终学到的参数
print("\n训练完成！")

print("---" * 10)
# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = model.forward(X)
print("y_numpy：", y_numpy)
print("预测结果：", y_predicted.shape)

plt.figure(figsize=(10, 6))
plt.scatter(x_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.scatter(x_numpy, y_predicted, label=f'Model: 多层网络输出', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
