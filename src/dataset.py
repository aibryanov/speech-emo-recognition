import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, path_to_data: str = "data/AudioMNIST", split: str = "train"):
        dataset_dict = load_from_disk(path_to_data)
        self.audioMNIST = dataset_dict[split]
        self.labels = self.audioMNIST["digit"]

    def __getitem__(self, index):
        row = self.audioMNIST[index]
        wav = torch.tensor(row["audio"]["array"], dtype=torch.float32)
        label = torch.tensor(row["digit"], dtype=torch.long)

        return wav, label

    def __len__(self):
        return len(self.audioMNIST)
