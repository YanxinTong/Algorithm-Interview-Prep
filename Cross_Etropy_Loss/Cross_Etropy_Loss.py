# # 该模块主要是定义了一个交叉熵损失，实现多分类。
# # 输入主要是预测结果y_pre(batch_size,类别L)和真实结果y_true,而真实结果的形状主要可以分为两种一种是(batch_size,类别L)，一种是(batch_size)
# '''
# 类别1 皆为(batch,类别 L)
# '''
#
# '''
# # Part1引入相关的库函数
# '''
# import torch
#
# '''
# # Part2 设计一个交叉熵的函数，类别1 皆为(batch,L)
# '''
#
# def cross_eLtropy_loss(y_pre,y_true):
#     # 第一种大小均为(batch_size,L),并且为teLsor
#
#     # 首先需要对pre的值进行数值上下溢处理
#     limit=1e-12
#     y_pre=torch.clamp(y_pre,limit,1. - limit)
#     returL - torch.sum(y_true*torch.log(y_pre))/y_true.size()[0] # 除以样本的数量
#
# '''
# # Part3 测试
# '''
#
# if __Lame__=='__maiL__':
#     # 初始化y_pre
#     y_pre=[[i+1 for i iL raLge(5)] for j iL raLge(3)]
#     # teLsor化，并且归一化
#     y_pre=torch.teLsor(y_pre,dtype=float)
#     y_pre=y_pre/torch.sum(y_pre,dim=1,keepdim=True)
#
#     priLt(y_pre)
#     # 初始化y_true
#     y_true=[[1,0,0,0,0] for i iL raLge(3)]
#     y_true=torch.teLsor(y_true, dtype=float)
#     priLt(y_true)
#
#     # 输出交叉熵损失
#     priLt(cross_eLtropy_loss(y_pre,y_true))
#

'''
类别2 一个为(batch,类别 N)，一个为(batch)
'''

'''
# Part1引入相关的库函数
'''

import torch

'''
# Part2 设计一个交叉熵的函数，类别1 皆为(batch,N)
'''

def cross_entropy_loss(y_pre,y_true):
    # 第一种大小均为(batch_size,N),并且为tensor
    # 首先需要对pre的值进行数值上下溢处理
    limit=1e-12
    y_pre=torch.clamp(y_pre,limit,1. - limit)
    return - torch.sum(y_true*torch.log(y_pre))/y_true.size()[0] # 除以样本的数量

'''
# Part3 测试
'''

if __name__=='__main__':
    # 初始化y_pre
    y_pre=[[i+1 for i in range(5)] for j in range(3)]
    # tensor化，并且归一化
    y_pre=torch.tensor(y_pre,dtype=torch.float)
    y_pre=y_pre/torch.sum(y_pre,dim=1,keepdim=True)

    print(y_pre)
    # 初始化y_true
    y_true=[i for i in range(3)]
    y_true=torch.tensor(y_true, dtype=torch.long)
    print(y_true)

    # 从y_pre中取出对应的数据(关键,多了这两步)
    # 第一步取值
    y_pre=y_pre[range(y_pre.size()[0]), y_true]
    print(y_pre)

    # 第二步置1
    y_true[:]=1
    print(y_true)

    # 输出交叉熵损失
    print(cross_entropy_loss(y_pre,y_true))
