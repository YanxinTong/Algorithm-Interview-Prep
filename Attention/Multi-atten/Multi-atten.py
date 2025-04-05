# 该模块实现的是多头注意力机制，和单头不一样的点
# 1. 需要把头提取出来，2. 需要对mask进行expand

'''
# Part1 引入相关的库函数
'''
import torch
from torch import nn
import math

'''
# Part 2 设计一个多头注意力的类
'''
class Multi_atten(nn.Module):
    def __init__(self,emd_size,q_k_size,v_size,head):
        super(Multi_atten,self).__init__()
        # 输入的x为(batch_size,seq_len,emd_size)
        # 第一步初始化三个全连接矩阵和头的数量
        self.head=head
        # 初始化是head的倍数，便于提取
        self.Wk=nn.Linear(emd_size,q_k_size*head)
        self.Wq=nn.Linear(emd_size,q_k_size*head)
        self.Wv=nn.Linear(emd_size,v_size*head)

        # 初始化Softmax函数
        self.softmax=nn.Softmax(dim=-1)

        # 剩下的等会看看
    def forward(self,x_q,x_k_v,mask):

        # 首先得到kvq
        q=self.Wq(x_q) # (batch_size,q_seq_len,q_size*head)
        k=self.Wk(x_k_v)
        v=self.Wv(x_k_v)

        # 其次是把头分出来得到多头的kvq
        q=q.reshape(q.size()[0],q.size()[1],self.head,-1).transpose(1,2) # (batch_size,head,q_seq_len,q_size)
        k = k.reshape(k.size()[0], k.size()[1], self.head, -1).transpose(1,2)
        v = v.reshape(q.size()[0], v.size()[1], self.head, -1).transpose(1,2)


        # 把k进行转置
        k=k.transpose(2,3) # (batch_size,head,k_seq_len,q_size)

        q_k=self.softmax(torch.matmul(q,k)/math.sqrt(k.size()[2]))


        # 进行mask(batch,seq_q,seq_k)
        if mask is not None:
            mask.unsqueeze(1).expand(-1,self.head,-1,-1)
        q_k.masked_fill(mask,1e-9)

        # 和v相乘
        atten=torch.matmul(q_k,v) # (batch_size,head,k_seq_len,k_v_size)

        # 将其进行返回原来的尺寸
        atten.transpose(1,2) # (batch_size,k_seq_len,head,k_v_size)
        atten=atten.reshape(atten.size()[0],atten.size()[1],-1) # (batch_size, k_seq_len, head*k_v_size)

        return atten

if __name__ == '__main__':
    # 类别1 单头的自注意力机制
    # 初始化输入x(batch_size,seq_len,emding)
    batch_size = 1  # 批量也就是句子的数量
    emd_size = 128  # 一个token嵌入的维度
    seq_len = 5  # kqv源的token长度
    q_k_size = emd_size//8  # q和k的嵌入维度
    v_size = emd_size//8  # v的嵌入维度

    x = torch.rand(size=(batch_size, seq_len, emd_size), dtype=torch.float)

    self_atten = Multi_atten(emd_size=emd_size, q_k_size=q_k_size, v_size=v_size,head=8)
    # 初始化mask(batch,len_k,len_q)
    mask = torch.randn(size=(batch_size, seq_len, seq_len))
    mask = mask.bool()

    print('单头的自注意力结果', self_atten(x, x, mask).size())

    # 类别2 单头的交叉注意力机制
    # 初始化输入x(batch_size,seq_len,emding)
    batch_size = 1  # 批量也就是句子的数量
    emd_size = 128  # 一个token嵌入的维度
    q_seq_len = 5  # q源的token长度
    q_k_size = emd_size//8  # q和k的嵌入维度/head
    k_v_seq_len = 7  # k_v源的token长度
    v_size = emd_size//8  # v的嵌入维度/head
    head=8 # 头的数量

    x_q = torch.rand(size=(batch_size, q_seq_len, emd_size), dtype=torch.float)
    x_k_v = torch.rand(size=(batch_size, k_v_seq_len, emd_size), dtype=torch.float)

    cross_atten = Multi_atten(emd_size=emd_size, q_k_size=q_k_size, v_size=v_size,head=head)
    # 初始化mask(batch,len_k,len_q)
    mask = torch.randn(size=(batch_size, q_seq_len, k_v_seq_len))
    mask = mask.bool()

    print('单头的交叉注意力结果', cross_atten(x_q, x_k_v, mask).size())




