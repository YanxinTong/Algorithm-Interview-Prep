import torch

def auc(y_true, y_scores):
    # 1. 按照预测概率降序排列
    sorted_indices = torch.argsort(y_scores, descending=True)  # 返回排序后的索引
    y_true_sorted = y_true[sorted_indices]  # 根据排序后的索引对y_true排序

    # 2. 统计正负样本总数
    total_positives = torch.sum(y_true_sorted)  # 正样本总数
    total_negatives = len(y_true_sorted) - total_positives  # 负样本总数
    print(total_negatives,total_positives)

    # 3. 对于负样本，累计它前面正样本的个数
    n_cum = torch.cumsum(y_true_sorted, dim=0)  # 累加正样本数量
    print(n_cum)
    auc_contribution = n_cum[y_true_sorted == 0]  # 对负样本贡献
    print(auc_contribution)

    # 4. 计算AUC
    auc = torch.sum(auc_contribution) / (total_negatives * total_positives)
    return auc

# 示例
y_true = torch.tensor([0, 0, 1, 1])  # 真实标签
y_scores = torch.tensor([0.1, 0.4, 0.35, 0.8])  # 预测分数

auc_value = auc(y_true, y_scores)
print(auc_value.item())
