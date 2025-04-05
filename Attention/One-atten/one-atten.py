# 该模块主要实现单头的注意力机制，输入为x，形成qkv，然后得到注意力z

'''
# Part1 引入相关的库函数
'''
import torch
from torch import nn
import math

'''
# Part2 定义单头注意力类
'''


class Onehead_Atten(nn.Module):
    def __init__(self, emd_size, q_k_size, v_size):
        super(Onehead_Atten, self).__init__()

        # Part1 初始化矩阵Wk,Wv,Wq
        # 注意x为(batch_size,q_seq_len,emd_size),且要实现(Q*KT)所以，Q(q_len,q_k_size),K为(k_len,q_k_size)
        self.Wk = nn.Linear(emd_size, q_k_size)
        self.Wq = nn.Linear(emd_size, q_k_size)
        self.Wv = nn.Linear(emd_size, v_size)

        # Part2 得到矩阵Q(batch_size,q_len,q_k_size),K(batch_size,k_v_len,q_k_size),V(batch_size,k_v_len,q_k_size)
        # softmax((Q*KT/sqrt(dk)))*V
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_q, x_k_v, mask=None):
        q = self.Wq(x_q)  # (batch_size,q_len,q_k_size)
        k = self.Wk(x_k_v)  # (batch_size,k_v_len,q_k_size)
        v = self.Wv(x_k_v)  # (batch_size,k_v_len,v_size)

        # 为了便于相乘对K转置
        k = k.transpose(1, 2)

        # 第一步把(Q*Kt)/根号dk
        q_k = self.softmax(torch.matmul(q, k) / math.sqrt(q.size()[-1]))


        # 判断是够要mask(1,seq_len_q,seq_len_k)
        if mask is not None:
            q_k = q_k.masked_fill(mask, 1e-9)

        # 第二步和v相乘
        atten_z = torch.matmul(q_k, v)
        return atten_z



if __name__ == '__main__':
    # 类别1 单头的自注意力机制
    # 初始化输入x(batch_size,seq_len,emding)
    batch_size = 1  # 批量也就是句子的数量
    emd_size = 128  # 一个token嵌入的维度
    seq_len = 5  # kqv源的token长度
    q_k_size = 128  # q和k的嵌入维度
    v_size = 128  # v的嵌入维度

    x = torch.rand(size=(batch_size, seq_len, emd_size), dtype=torch.float)

    self_atten = Onehead_Atten(emd_size=emd_size, q_k_size=q_k_size, v_size=v_size)
    # 初始化mask(batch,len_k,len_q)
    mask = torch.randn(size=(batch_size, seq_len, seq_len))
    mask = mask.bool()

    print('单头的自注意力结果',self_atten(x, x, mask).size())

    # 类别2 单头的交叉注意力机制
    # 初始化输入x(batch_size,seq_len,emding)
    batch_size=1 # 批量也就是句子的数量
    emd_size=128 # 一个token嵌入的维度
    q_seq_len=5 # q源的token长度
    q_k_size=128 # q和k的嵌入维度/head
    k_v_seq_len=7 # k_v源的token长度
    v_size=128  # v的嵌入维度/head
    x_q = torch.rand(size=(batch_size, q_seq_len, emd_size), dtype=torch.float)
    x_k_v = torch.rand(size=(batch_size, k_v_seq_len, emd_size), dtype=torch.float)

    cross_atten = Onehead_Atten(emd_size=emd_size, q_k_size=q_k_size, v_size=v_size)
    # 初始化mask(batch,len_k,len_q)
    mask = torch.randn(size=(batch_size, q_seq_len, k_v_seq_len))
    mask = mask.bool()

    print('单头的交叉注意力结果',cross_atten(x_q, x_k_v, mask).size())