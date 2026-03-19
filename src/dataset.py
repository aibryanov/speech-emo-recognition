from pathlib import Path

from torch.utils.data import Dataset
import torch
import librosa

from src.hf_dataset import hf_audioMNIST


audioMNIST = hf_audioMNIST.with_format("torch")

class MNISTDataset(Dataset):
    def __init__(self, path_to_data):
        self.paths = list(map(str, Path(path_to_data).glob('**/*.wav')))
        self.labels = list(map(lambda x: int(Path(x).name.split('_')[0]), self.paths))

    def __getitem__(self, index):
        wav, sr = librosa.load(self.paths[index])
        label = self.labels[index]

        return torch.tensor(wav, dtype=torch.float), label

    def __len__(self):
        return len(self.paths)