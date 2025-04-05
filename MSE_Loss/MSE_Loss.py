# 本模块主要是为了实现MSE损失

'''
# Part1 引入相关的库函数
'''

import torch

'''
# 定义MSE损失
'''


def MSE_Loss(y_pre, y_true):  # 均为(batch_size,image_size, image_size)
    """
        手动实现均方误差 (MSE) 损失函数

        参数:
        y_pre: 模型的预测值 (Tensor)
        y_true: 真实值 (Tensor)

        返回:
        平均 MSE 损失
    """
    # 先相减，然后平方，最后取均值。
    delta = y_pre - y_true
    delta_2 = (delta) ** 2
    return torch.mean(delta_2)


'''
# 测试
'''
if __name__ == '__main__':
    y_pre = [[[i for i in range(3)] for j in range(3)] for batch_size in range(2)] # (2,3,3)
    y_pre = torch.tensor(y_pre, dtype=float)

    y_true = [[[i + 1 for i in range(3)] for j in range(3)] for batch_size in range(2)] # (2,3,3)
    y_true = torch.tensor(y_true, dtype=float)

    print(MSE_Loss(y_pre, y_true))
