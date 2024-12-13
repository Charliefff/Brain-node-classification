import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataloader import CNN_Dataset

def load_tensor(file_path):
    return torch.load(file_path)

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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_data_with_dataloader(config):
    train_E = load_tensor(config["train_E"])
    train_W = load_tensor(config["train_W"])
    test_E = load_tensor(config["test_E"])
    test_W = load_tensor(config["test_W"])

    train_data, train_labels = concatenate_tensors(train_E, train_W, labels=True)
    test_data, test_labels = concatenate_tensors(test_E, test_W, labels=True)

    train_data, val_data, train_labels, val_labels = split_data(train_data, train_labels)

    training_dataset = CNN_Dataset(train_data, train_labels)
    validation_dataset = CNN_Dataset(val_data, val_labels)
    testing_dataset = CNN_Dataset(test_data, test_labels)

    train_loader = create_dataloader(training_dataset, config["batch_size"], shuffle=True)
    validate_loader = create_dataloader(validation_dataset, config["batch_size"], shuffle=False)
    test_loader = create_dataloader(testing_dataset, config["batch_size"], shuffle=False)

    return train_loader, validate_loader, test_loader
