import torch
def l1_regularization(weights, lambda_l1=1e-4):
    reg_loss = lambda_l1 * torch.norm(weights, p=1)  # L1 範數
    return reg_loss

def temporal_lasso(wF, lambda_tl=1e-5):
    diff = wF[:, 1:] - wF[:, :-1]  # 計算相鄰時間步差異
    reg_loss = lambda_tl * torch.norm(diff, p=1)  # L1 範數
    return reg_loss

def sparse_group_lasso(weights, group_indices, lambda_gl=1e-4, lambda_l1=1e-4):
    reg_loss = 0.0
    for group in group_indices:
        group_weights = weights[..., group]  # 選擇對應組的權重
        reg_loss += lambda_gl * torch.sqrt(torch.tensor(group_weights.numel(), dtype=torch.float32)) * torch.norm(group_weights, p=2)
    reg_loss += lambda_l1 * torch.norm(weights, p=1)  # L1 範數
    return reg_loss
