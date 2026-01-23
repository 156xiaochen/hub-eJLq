import torch
import torch.nn as nn
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成模拟数据
x_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
# 形状为 (200,1) 的二维数组，其中包含 200 个在 [0, 2π] 范围内均匀分布的随机浮点数。
y_numpy = np.sin(x_numpy) + 0.1 * np.random.randn(*x_numpy.shape)

x = torch.from_numpy(x_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络模型
class SinFitter(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1):
        """
        多层感知机（MLP）：用于拟合sin函数的非线性模型
        :param input_dim: 输入维度（x是一维，所以=1）
        :param hidden_dims: 隐藏层维度列表（多层）
        :param output_dim: 输出维度（y是一维，所以=1）
        """
        super(SinFitter, self).__init__()
        layers = []
        # 输入层 → 第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())  # 非线性激活函数（关键！拟合非线性sin曲线）

        # 隐藏层之间的连接（多层）
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())

        # 最后一个隐藏层 → 输出层（回归任务，无激活函数）
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # 组合所有层
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 初始化模型（2层隐藏层：64→32节点）
model = SinFitter(input_dim=1, hidden_dims=[64, 32], output_dim=1)
print("模型结构：")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务

# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器（比SGD更适合非线性拟合）

# 4. 训练模型
num_epochs = 2200
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
with torch.no_grad():
    y_pred = model(x).numpy()

plt.figure(figsize=(12, 6))
plt.scatter(x_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(x_numpy, y_pred, label='sin(x)', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y = sin(x)')
plt.legend()
plt.grid(True)
plt.show()
