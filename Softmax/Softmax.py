import torch


# def softmax(X): # X为Tensor向量，大小为(batch_size,len)
#     # # 方法一，复杂版本
#     # for i in range(X.size()[0]):
#     #     # 取出某行的Tensor
#     #     # 为了防止数据的上下，先把数据减去最大值
#     #     X[i]-=max(X[i].clone())
#     #     X[i]=torch.exp(X[i])
#     #     X[i]/=X[i].sum()
#     # return X
#
#     # 方法二：简单版本
#
#     # 增加一步，防止数据上下溢出
#     # (batch_size, 1)
#     X_max,X_index=torch.max(X,dim=1,keepdim=True) # 让其保持二维
#     X -= X_max
#     # 取exp
#     X_exp = torch.exp(X)
#     # 求和从1维求和得到的是(batch_size,1)
#     X_sum=X_exp.sum(dim=1,keepdim=True)
#     return X_exp/X_sum

# 第一次训练
def softmax(X):  # X(batch,size)
    max_x, _ = torch.max(X, dim=-1, keepdim=True) # 只要torch涉及到sum,max等等,都是需要keepdim的
    X -= max_x
    X = torch.exp(X)
    X /= torch.sum(X, dim=-1, keepdim=True)
    return X


if __name__ == '__main__':
    X = [[i for i in range(4)], [j for j in range(1, 5)], [5, 4, 3, 2]]

    X = torch.tensor(X, dtype=torch.float)
    print(softmax(X))
