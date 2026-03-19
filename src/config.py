from dataclasses import dataclass, field
from enum import Enum

from hydra.core.config_store import ConfigStore


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class DatasetConfig:
    name: str = "audio_mnist"
    path_to_data: str = "data/AudioMNIST"
    file_pattern: str = "**/*.wav"
    sample_rate: int = 16_000


@dataclass
class DataloaderConfig:
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 10
    learning_rate: float = 1e-3
    device: Device = Device.CPU


@dataclass
class AppConfig:
    experiment_name: str = "audio_mnist_baseline"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=AppConfig)
    cs.store(group="dataset", name="audio_mnist", node=DatasetConfig)
