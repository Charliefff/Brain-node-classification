import torch
import pandas as pd


# 載入數據
def load_data(train_E_path, train_W_path, test_E_path, test_W_path, adj_path=None):
    # 載入數據
    tensor_train_E = torch.load(train_E_path)
    tensor_train_W = torch.load(train_W_path)
    tensor_test_E = torch.load(test_E_path)
    tensor_test_W = torch.load(test_W_path)
    
    # 正規化數據
    tensor_train_E, tensor_test_E = normalize(tensor_train_E, tensor_test_E)
    tensor_train_W, tensor_test_W = normalize(tensor_train_W, tensor_test_W)
    
    # 合併數據
    train_data = torch.cat((tensor_train_E, tensor_train_W), dim=0)
    test_data = torch.cat((tensor_test_E, tensor_test_W), dim=0)

    # 生成標籤
    train_label = torch.cat(
        (torch.zeros(tensor_train_E.size(0)), torch.ones(tensor_train_W.size(0))),
        dim=0).long()
    test_label = torch.cat(
        (torch.zeros(tensor_test_E.size(0)), torch.ones(tensor_test_W.size(0))),
        dim=0).long()

    # [sample, feature, node] -> [sample, node, feature]
    train_data = train_data.transpose(1, 2)
    test_data = test_data.transpose(1, 2)
    
    # 處理鄰接矩陣
    if adj_path is None:
        num_nodes = train_data.size(1)  # 假設所有數據的節點數一致
        df_adj = torch.ones((num_nodes, num_nodes), dtype=torch.float32)
    else:
        df = pd.read_csv(adj_path, index_col=0)
        df_adj = torch.tensor(df.values, dtype=torch.float32)

    # 返回所有數據
    return train_data, test_data, train_label, test_label, df_adj


# 正規化數據
def normalize(train_data, test_data):
    train_data_mean = train_data.mean(dim=(0, 1), keepdim=True)
    train_data_std = train_data.std(dim=(0, 1), keepdim=True)

    train_data = (train_data - train_data_mean) / train_data_std
    test_data = (test_data - train_data_mean) / train_data_std

    return train_data, test_data


# 擴增鄰接矩陣
def densify_adj(adj_sparse, add_edges_ratio=0.1):
    adj_sparse = adj_sparse.to_sparse()
    num_nodes = adj_sparse.size(0)
    num_add_edges = int(num_nodes * num_nodes * add_edges_ratio)
    
    indices = adj_sparse._indices()
    mask = torch.ones((num_nodes, num_nodes), device=adj_sparse.device)
    mask[indices[0], indices[1]] = 0 
    
    zero_indices = torch.nonzero(mask, as_tuple=False).T
    sampled_indices = zero_indices[:, torch.randint(0, zero_indices.size(1), (num_add_edges,))]

    new_values = torch.ones(sampled_indices.size(1), device=adj_sparse.device)
    new_indices = torch.cat([indices, sampled_indices], dim=1)
    new_values = torch.cat([adj_sparse._values(), new_values], dim=0)
    
    return torch.sparse_coo_tensor(new_indices, new_values, adj_sparse.size())

