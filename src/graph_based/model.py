import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, apply_relu=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.apply_relu = apply_relu

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.size()

        x_out = []
        for i in range(batch_size):
            adj_single = adj[i]
            adj_hat = adj_single + torch.eye(num_nodes, device=adj.device)
            degree = torch.sum(adj_hat, dim=1)
            degree = torch.pow(degree, -0.5)
            degree[degree == float('inf')] = 0.0  # 避免無窮大
            degree_matrix = torch.diag(degree)
            normalized_adj = degree_matrix @ adj_hat @ degree_matrix
            x_out.append(normalized_adj @ x[i])

        x_out = torch.stack(x_out)
        x_out = self.dropout(x_out)
        if self.apply_relu:
            return torch.relu(self.linear(x_out))
        else:
            return self.linear(x_out)

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_features, hidden_features, dropout))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_features, hidden_features, dropout))

        self.layers.append(GCNLayer(hidden_features, out_features, dropout=0.0, apply_relu=False))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = layer(x, adj)
        x = self.layers[-1](x, adj)
        return x

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.0, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

        # 將輸入特徵線性映射到多頭輸出特徵空間
        self.nn = nn.Linear(in_features, num_heads * out_features, bias=False)

        # 注意力參數: [num_heads, 2*out_features]
        self.attention = nn.Parameter(torch.empty(size=(num_heads, 2 * out_features)))
        nn.init.xavier_uniform_(self.attention.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.concat = concat

    def forward(self, x, adj):
        # x: [batch_size, num_nodes, in_features]
        batch_size, num_nodes, _ = x.size()

        # 特徵投影至多頭空間: [b, n, h*d]
        x_proj = self.nn(x)  
        x_proj = x_proj.view(batch_size, num_nodes, self.num_heads, self.out_features)  
        # 現在 x_proj: [b, n, h, d]

        # 為了計算 (i, j) 節點對的注意力分數，準備特徵:
        # x_proj_i: [b, i, j, h, d] (將每個節點特徵複製擴展到 j 維)
        x_proj_i = x_proj.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        # x_proj_j: [b, i, j, h, d] (將每個節點特徵複製擴展到 i 維)
        x_proj_j = x_proj.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)
        
        # 合併 i 和 j 節點特徵: x_concat: [b, i, j, h, 2*d]
        x_concat = torch.cat([x_proj_i, x_proj_j], dim=-1)

        # 利用注意力參數計算注意力分數: 
        # x_concat: [b, i, j, h, 2*d]
        # self.attention: [h, 2*d]
        # einsum("bijhk,hk->bhij") 
        # 解釋:
        #   bijhk 中的 h,k 維度和 hk 中的 h,k 相乘並對 k 進行求和，
        #   保留 b,h,i,j 四個維度 => [b, h, i, j]
        attention_logits = torch.einsum("bijhk,hk->bhij", x_concat, self.attention)
        attention_logits = self.leaky_relu(attention_logits)

        # adj: [b, n, n]
        # 將 adj 擴張至 [b, h, n, n] 以匹配多頭注意力維度
        mask = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # 對非鄰接邊緣使用 -inf，確保 softmax 後為零
        attention_logits = attention_logits.masked_fill(mask == 0, float('-inf'))

        # 對鄰接節點維度 (j) 做 softmax 
        attention = F.softmax(attention_logits, dim=-1)
        attention = self.dropout(attention)  # dropout

        # 將 x_proj 調整為 [b, h, n, d] 以便與 attention 相乘
        x_proj = x_proj.permute(0, 2, 1, 3)
        # attention: [b, h, i, j]
        # x_proj   : [b, h, j, d]
        # einsum("bhij,bhjd->bhid") 對 j 進行加權求和得到 [b, h, i, d]
        h_prime = torch.einsum("bhij,bhjd->bhid", attention, x_proj)

        # 將多頭輸出拼接回 [b, n, h*d]
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, -1)
        
        # concat multi-heads
        if self.concat:
            h_prime = h_prime.view(batch_size, num_nodes, -1)
        else:
            h_prime = h_prime.mean(dim=1, keepdim=False)
        return h_prime

class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads=4, dropout=0.5, num_layers=2):
        super(GAT, self).__init__()
        assert num_layers >= 2, "num_layers should be at least 2"
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_features, hidden_features, num_heads=num_heads, dropout=dropout))

        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden_features * num_heads, hidden_features, num_heads=num_heads, dropout=dropout))
            
        self.layers.append(GATLayer(hidden_features * num_heads, out_features, num_heads=1, dropout=0.0))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = layer(x, adj)
            x = F.elu(x)
        x = self.layers[-1](x, adj)
        return x

class GAT_GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads=4, dropout=0.5, num_layers=2):
        super(GAT_GCN, self).__init__()
        self.gcn = GCN(in_features, hidden_features, hidden_features, num_layers, dropout)
        self.gat = GAT(hidden_features, hidden_features, out_features, num_heads, dropout, num_layers)

    def forward(self, x, adj):
        x = self.gcn(x, adj)
        x = F.elu(x)
        x = self.gat(x, adj)
        return x

class Model(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 out_features, 
                 num_layers=2, 
                 dropout=0.5, 
                 num_heads=4, 
                 model_type="gcn"):
        super(Model, self).__init__()
        if model_type == "gcn":
            print("Using GCN model")
            self.model = GCN(in_features, hidden_features, out_features, num_layers, dropout)
        elif model_type == "gat":
            print("Using GAT model")
            self.model = GAT(in_features, hidden_features, out_features, num_heads, dropout, num_layers)
        elif model_type == "gcn_gat":
            print("Using GAT-GCN model")
            self.model = GAT_GCN(in_features, hidden_features, out_features, num_heads, dropout, num_layers)
        else:
            raise ValueError("Invalid Input model_type. Must be one of ['gcn', 'gat', 'gcn_gat']")

    def forward(self, x, adj):
        return self.model(x, adj)

if __name__ == "__main__":

    x = torch.randn(2, 4, 8)

    adj = torch.randint(0, 2, (2, 4, 4))
    adj = (adj + adj.transpose(1,2)) > 0
    adj = adj.int()

    model = GAT(in_features=8, 
                hidden_features=16, 
                out_features=3, 
                num_heads=4, 
                dropout=0.1, 
                num_layers=2)
    out = model(x, adj)
    print("GAT output shape:", out.shape) 
