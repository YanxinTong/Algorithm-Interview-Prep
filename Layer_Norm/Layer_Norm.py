# 本模块主要是为了实现LayerNorm模块

'''
# 第一步，引入相关的库函数，因为该模块属于网络部分，所以需要继承网络模块
'''

import torch
from torch import nn

'''
# 第二步，初始化LayerNrom函数，输入为层归一化的维度(注意要转化为元组的类型)，是否需要可学习的伽马和β参数
'''


# class LayerNorm(nn.Module):
#     def __init__(self, norm_size, limit, is_learn_p=False):
#         # 确定LayerNorm为子类，继承Module模块
#         super(LayerNorm, self).__init__()
#
#         self.is_learn_p = is_learn_p
#         # 判断为单维还是多维，无论是单维度还是多维，都转化为tuple类型
#         self.norm_size = (norm_size,) if isinstance(norm_size, int) else tuple(norm_size)
#
#         # 初始化防止上下溢的参数，主要是在除数(方差)里面不要让分母为0
#         self.limit = limit
#
#         # 初始化需要学习的参数，主要是用于对于均值的平移u=u*γ+bata
#         if is_learn_p:
#             self.gamma = nn.Parameter(torch.ones(size=self.norm_size))
#             self.beta = nn.Parameter(torch.zeros(size=self.norm_size))
#         else:
#             self.gamma = None
#             self.beta = None
#
#     def forward(self, x):
#         # 步骤是对需要归一化的维度进行取均值和方差，然后减去均值和除以根号方尺，最后判断是否需要学习参数，把参数给添加上。
#
#         # 首先判断需要归一化的维度是哪几维，一般要么一维度，要么二维。
#         dim = (-2, -1) if len(self.norm_size) == 1 else -1
#
#         # 取均值和方差
#         mean = torch.mean(x, dim=dim, keepdim=True)
#         var = torch.var(x, dim=dim, keepdim=True)
#
#         # 对利用广播机制进行减去和加上
#         x_norm = (x - mean) / torch.sqrt((var + self.limit))
#
#         # 判断是否由学习参数
#         if self.is_learn_p:
#             x_norm = x_norm * self.gamma + self.beta
#
#         return x_norm

# 第一次训练
class Layer_norm(nn.Module):
    def __init__(self, norm_size,limit):
        super().__init__()
        # 输入的话,可能是(batch,seq_len,emd_size),(batch,channel,h,w)
        self.norm_size=(norm_size,) if isinstance(norm_size,int) else tuple(norm_size)
        self.limit=limit
        #
    def foward(self,x):
        dim=(-2,-1) if len(self.norm_size)==2 else -1
        u=torch.mean(x,dim=dim,keepdim=True)
        sigma=torch.sqrt(torch.var(x,dim=dim))
        x=(x-u)/(sigma+self.limit)
        return x

'''
# 第三步 测试，初始化参数
'''

if __name__ == '__main__':
    # 第一种，为图像类型，层归一化的是最后图像层
    x = torch.randn(4, 3, 25, 25)

    layernorm = LayerNorm(norm_size=(25, 25), limit=1e-12, is_learn_p=False)

    output = layernorm(x)

    print("图像输入形状:", x.size())
    print("图像输出形状:", output.size())

    # 第二种，是单维，比如seq，维度为(batch, seq_len, emd_size)
    x = torch.randn(3, 25, 25)

    layernorm = LayerNorm(norm_size=25, limit=1e-12, is_learn_p=False)

    output = layernorm(x)

    print("语言输入形状:", x.size())
    print("语言输出形状:", output.size())
