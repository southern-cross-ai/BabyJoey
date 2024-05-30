import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        # Initialize your dataset, e.g., load data from files
        pass

    def __len__(self):
        # Return the length of your dataset
        pass

    def __getitem__(self, idx):
        # Return a single data point as a tensor
        pass

def get_dataloader(config):
    dataset = CustomDataset(config['data_path'])
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
