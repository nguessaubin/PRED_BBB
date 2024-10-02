import data_setup
import torch
from torch.utils.data import Dataset, DataLoader

# Create Custom Dataset for Dataloader 

class BBBDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# Get X and y
def Custum_dataloader(types: str,
            batch_size: int,
            num_workers:int):
    X,y =data_setup.data_setup("seyonec/ChemBERTa-zinc-base-v1","seyonec/ChemBERTa-zinc-base-v1",types)
    y= torch.Tensor(y).unsqueeze(dim=1)

    dataset = BBBDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader