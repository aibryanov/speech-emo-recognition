import torch
import torchaudio
from datasets import load_from_disk, load_dataset
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(
        self,
        path_to_data: str = "data/AudioMNIST",
        split: str = "train",
        target_sample_rate: int | None = None,
    ):
        dataset_dict = load_from_disk(path_to_data)
        self.audioMNIST = dataset_dict[split]
        self.labels = self.audioMNIST["digit"]
        self.class_names = [str(i) for i in sorted(set(self.labels))]
        self.target_sample_rate = target_sample_rate
        self._resamplers = {}

    def _downsample_if_needed(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self.target_sample_rate is None or sample_rate <= self.target_sample_rate:
            return wav

        if sample_rate not in self._resamplers:
            self._resamplers[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )

        return self._resamplers[sample_rate](wav)

    def __getitem__(self, index):
        row = self.audioMNIST[index]
        wav = torch.tensor(row["audio"]["array"], dtype=torch.float32)
        sample_rate = int(row["audio"]["sampling_rate"])
        wav = self._downsample_if_needed(wav, sample_rate)
        label = torch.tensor(row["digit"], dtype=torch.long)

        return wav, label

    def __len__(self):
        return len(self.audioMNIST)


class RESDDataset(Dataset):
    def __init__(self, split="train", target_sample_rate: int | None = None):
        self.dataset = load_dataset("Aniemore/resd")[split]
        self.mapping = {
            "fear": 0,
            "anger": 1, 
            "happiness": 2,
            "enthusiasm": 3, 
            "neutral": 4,
            "disgust": 5,
            "sadness": 6,
        }
        self.class_names = [None] * len(self.mapping)
        for label_name, label_id in self.mapping.items():
            self.class_names[label_id] = label_name
        self.target_sample_rate = target_sample_rate
        self._resamplers = {}

    def _downsample_if_needed(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self.target_sample_rate is None or sample_rate <= self.target_sample_rate:
            return wav

        if sample_rate not in self._resamplers:
            self._resamplers[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )

        return self._resamplers[sample_rate](wav)

    def __getitem__(self, index):
        row = self.dataset[index]
        wav = torch.tensor(row["speech"]["array"], dtype=torch.float32)
        sample_rate = int(row["speech"]["sampling_rate"])
        wav = self._downsample_if_needed(wav, sample_rate)
        emo = str(row["emotion"]).strip()

        label = torch.tensor(self.mapping[emo], dtype=torch.long)

        return wav, label
    
    def __len__(self):
        return len(self.dataset)
