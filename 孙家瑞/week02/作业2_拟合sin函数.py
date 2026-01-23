import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# 1. 生成模拟数据 (与之前相同)

def gen_dataset(count=2000, low=-10, high=10):
    X = np.random.uniform(low, high, size=(count, 1))
    Y = np.sin(X) + np.random.normal(loc=0.1, scale=0.05, size=X.shape)
    return torch.FloatTensor(np.stack(X)), torch.FloatTensor(np.stack(Y))


# 2. 定义一个模型
class FitSin(nn.Module):
    def __init__(self, hidden_size=100, in_f=100):
        super(FitSin, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.loss = nn.MSELoss()

    def forward(self, x, y_true=None):
        y_pred = self.network(x)
        if y_true is not None:
            return self.loss(y_pred, y_true)
        else:
            return y_pred


# 3. 训练模型
def train_model(model_shape=512, dataset=gen_dataset(), epoch_size=100, learning_rate=0.01, batch_size=100):
    model = FitSin(model_shape, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 获取训练集数据
    train_X, train_Y = dataset
    model.train()
    for epoch in range(epoch_size):
        loss_change = []
        for b_idx in range(train_X.shape[0] // batch_size):
            x = train_X[b_idx * batch_size:(b_idx + 1) * batch_size]
            y = train_Y[b_idx * batch_size:(b_idx + 1) * batch_size]
            loss = model.forward(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_change.append(loss.item())
        print(f"epoch {epoch}, train loss: {np.mean(loss_change)}")
        # 损失函数比较小的时候，直接终止训练
        if (np.mean(loss_change) < 0.005):
            break
    return model


# 4. 绘制结果
def plot_graph_of_fit(model: FitSin, dataset=gen_dataset()):
    X, Y = dataset
    model.eval()
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, label='Raw data', color='blue', alpha=0.01)
    with torch.no_grad():
        plt.scatter(X, model(X), label=f'Model: y = sin(x)', color='red', alpha=0.1)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_graph_of_fit(train_model())
