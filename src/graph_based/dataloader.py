from torch.utils.data import DataLoader, Dataset

class GCNDataset(Dataset):
    
    def __init__(self, data, labels, adj):
  
        self.data = data
        self.labels = labels
        self.adj = adj

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.adj


def create_dataloader(data, labels, adj, batch_size=64, shuffle=True):

    dataset = GCNDataset(data, labels, adj)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)