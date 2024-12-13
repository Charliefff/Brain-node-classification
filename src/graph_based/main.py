import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from load_data import load_data, densify_adj
from dataloader import create_dataloader
from trainer import train_model, test_model
from model import Model

def main():
    # 讀取配置檔
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加載資料
    train_data, test_data, train_labels, test_labels, adj_sparse = load_data(
        train_E_path=config["data"]["train_E"],
        train_W_path=config["data"]["train_W"],
        test_E_path=config["data"]["test_E"],
        test_W_path=config["data"]["test_W"],
        adj_path=config["data"]["adj"]
    )
    
    # 可選：截取特徵
    if config["data"]["cut_off"]:
        train_data = train_data[:, :, 125:]
        test_data = test_data[:, :, 125:]
    
    # 增強鄰接矩陣
    if config["data"]["adj"] is None:
        adj_sparse = adj_sparse.clone().detach().to(torch.float32).to(device)
    else:
        adj_sparse = densify_adj(adj_sparse, add_edges_ratio=config["data"]["add_edges_ratio"])
    
    # 訓練集與驗證集分割
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.1, random_state=42
    )
    print(f"Training data shape: {train_data.shape}, Validation data shape: {val_data.shape}")
    
    # 創建 DataLoader
    batch_size = config["training"]["batch_size"]
    train_loader = create_dataloader(train_data, train_labels, adj_sparse, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, val_labels, adj_sparse, batch_size, shuffle=False)
    test_loader = create_dataloader(test_data, test_labels, adj_sparse, batch_size, shuffle=False)

    # 初始化模型
    model = Model(
        in_features=config["model"]["in_features"],
        hidden_features=config["model"]["hidden_features"],
        out_features=config["model"]["out_features"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        num_heads=config["model"]["num_heads"],
        model_type=config["model"]["model_type"]
    )
   
    model.to(device)
    model.to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-3)

    # 訓練模型
    print("Starting training...")
    train_model(
        train_loader=train_loader, 
        val_loader=val_loader, 
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        num_epochs=config["training"]["num_epochs"], 
        device=device,
        log_dir=config["training"]["log_dir"]
    )

    # 測試模型
    print("Starting testing...")
    test_model(test_loader=test_loader, model=model, criterion=criterion, device=device)

if __name__ == "__main__":
    main()
