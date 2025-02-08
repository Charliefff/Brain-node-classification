import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse 

from model import modelType
from trainer import test_cnn, train_cnn
from load_data import load_data_with_dataloader

import warnings
import os
import random
import numpy as np

# 忽略 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_config(filepath='config.yaml') -> dict:
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)

    flat_config = {
        **config.get("training", {}),
        **config.get("data", {}),
        **config.get("model", {})
    }
    return flat_config


def read_params():
    parser = argparse.ArgumentParser(description="Training and testing a model with configurable parameters.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config YAML file.")
    args = parser.parse_args()
    return vars(args)

def printConfig(config):
    print("\n**** Configuration ****")

    # 打印分層結構
    print("\n[Data]")
    for key in ["train_E", "train_W", "test_E", "test_W", "log_dir", "save_path"]:
        if key in config:
            print(f"{key}: {config[key]}")

    print("\n[Training]")
    for key in ["seed", "init_weight","batch_size", "num_epochs", "learning_rate"]:
        if key in config:
            print(f"{key}: {config[key]}")

    print("\n[Model]")
    for key in ["model_type", "num_classes", "channels", "kernLength", "squeeze_dim", "F1", "F2", "poolKern1", "poolKern2", "dropoutRate", "dropoutType", "Dim_expand", "enable_atten"]:
        if key in config:
            print(f"{key}: {config[key]}")

    print("\n*********************\n")


def main():
    cli_params = read_params()
    config = read_config(filepath=cli_params["config"])
    set_seed(config["seed"])

    config.update({k: v for k, v in cli_params.items() if v is not None})    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)
    printConfig(config=config)
    
    train_loader, validate_loader, test_loader = load_data_with_dataloader(config)
    model = modelType(**config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    print("Starting training...")
    train_cnn(
        train_loader, 
        validate_loader, 
        test_loader,
        model, 
        optimizer, 
        criterion, 
        scheduler,
        num_epochs=config["num_epochs"], 
        device=device,
        log_dir=config["log_dir"],
        save_path=config["save_path"], 
        enable_atten=config["enable_atten"], 
        model_name = config["model_type"]
    )

    
    print("Starting testing...")
    test_cnn(test_loader, model, criterion, device=device)


if __name__ == "__main__":
    main()