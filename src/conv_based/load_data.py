import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataloader import CNN_Dataset

def load_tensor(file_path):
    return torch.load(file_path).float()

def concatenate_tensors(tensor1, tensor2, labels=False):
    combined = torch.cat((tensor1, tensor2), dim=0)
    if labels:
        label1 = torch.zeros(tensor1.size(0))
        label2 = torch.ones(tensor2.size(0))
        return combined, torch.cat((label1, label2), dim=0).long()
    return combined

def split_data(data, labels, val_size=0.1, random_state=42):
    return train_test_split(data, labels, test_size=val_size, random_state=random_state)

def create_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


def load_data_with_dataloader(config):
    # 載入資料
    # 加載張量並切片保留前 375
    train_E = load_tensor(config["train_E"])[:, :375, :]
    train_W = load_tensor(config["train_W"])[:, :375, :]
    test_E = load_tensor(config["test_E"])[:, :375, :]
    test_W = load_tensor(config["test_W"])[:, :375, :]

    train_data, train_labels = concatenate_tensors(train_E, train_W, labels=True)
    test_data, test_labels = concatenate_tensors(test_E, test_W, labels=True)

    train_data, val_data, train_labels, val_labels = split_data(train_data, train_labels)

    mean = train_data.mean(dim=(0, 1, 2), keepdim=True)  # 對 batch 和 375 64度計算均值
    std = train_data.std(dim=(0, 1, 2), keepdim=True)    # 對 batch 和 375 64度計算標準差

    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (test_data - mean) / std

    training_dataset = CNN_Dataset(train_data, train_labels)
    validation_dataset = CNN_Dataset(val_data, val_labels)
    testing_dataset = CNN_Dataset(test_data, test_labels)

    train_loader = create_dataloader(training_dataset, config["batch_size"], shuffle=True)
    validate_loader = create_dataloader(validation_dataset, config["batch_size"], shuffle=False)
    test_loader = create_dataloader(testing_dataset, config["batch_size"], shuffle=False)

    return train_loader, validate_loader, test_loader