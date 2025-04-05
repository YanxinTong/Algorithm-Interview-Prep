# 该模块主要是为了实现FashionMinist图像分类。图像的大小为(28,28),类别为同样为10类
'''
# Part1引入相关的库函数
'''
import torch
from torch import nn
from torch.utils import data

import torchvision
from torchvision import transforms

'''
# Part2 数据集的加载,和dataloader的初始化
'''

transforms_action = [transforms.ToTensor()]
transforms_action = transforms.Compose(transforms_action)

Minist_train = torchvision.datasets.FashionMNIST(root='Minist', train=True, transform=transforms_action, download=True)
Minist_test = torchvision.datasets.FashionMNIST(root='Minist', train=False, transform=transforms_action, download=True)

train_dataloader = data.DataLoader(dataset=Minist_train, batch_size=15, shuffle=True)
test_dataloader = data.DataLoader(dataset=Minist_test, batch_size=15, shuffle=True)

'''
# Part3 初始化各种前向传播需要的1. 网络 2. Loss 3. 优化器
'''


class MLP(nn.Module):
    def __init__(self, image_size, num_kind,latent=128):
        super(MLP, self).__init__()
        self.Linear1 = nn.Linear(image_size, latent, bias=False)

        self.relu1 = nn.ReLU()

        self.Linear2 = nn.Linear(latent, num_kind, bias=False)

        # 计算CrossEntropyLoss时候会自动计算softmax所以不需要
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # (batch,1,28,28)
        x = x.reshape(x.size()[0], -1)
        x = self.Linear1(x)
        x = self.relu1(x)
        x = self.Linear2(x)
        # x = self.softmax(x)
        return x  # (batch,10)


net = MLP(784, 10)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params=net.parameters(), lr=1e-3)

'''
# Part4 循环训练计算损失
'''

epochs = 10

for epoch in range(epochs):
    for images, labels in train_dataloader:
        # 首先前向传播
        result = net(images)

        # 计算损失
        L = loss(result, labels)

        # 反向传播
        L.backward()

        # 参数更新
        optimizer.step()

        # 清除梯度
        optimizer.zero_grad()

    # 存储模型
    torch.save(net, 'checkpoint/module_epoch_{}.pth'.format(epoch))

    # 每个epoch在测试集跑一遍进行计算平均损失
    total_loss = 0
    total_batches = 0
    with torch.no_grad():
        for images_test, labels_test in Minist_test:
            # 形状是Batchsize*hang
            labels_hat = net(images_test)
            L_test = loss(labels_hat, labels_test)
            total_loss += L_test.item()
            total_batches += 1

            # 计算平均测试损失并记录
        avg_test_loss = total_loss / total_batches
        print(f'第 {epoch + 1} 轮训练完成，平均测试损失为：{avg_test_loss}')
