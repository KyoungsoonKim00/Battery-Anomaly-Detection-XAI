import torch
from torch.utils.data import Dataset

class BatteryDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]