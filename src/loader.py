import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from src.dataset import MNISTDataset, RESDDataset


def zero_pad_collate_fn(batch):
    wavs, labels = zip(*batch)

    wavs = pad_sequence(wavs, batch_first=True)
    labels = torch.stack(labels)

    return wavs, labels

def train_test_split(
        cfg,
        dataset: Dataset,
        test_ratio: float = 0.2,
):  
    generator = torch.Generator().manual_seed(cfg.seed)

    train_size = int(len(dataset) * (1 - test_ratio))
    test_size = len(dataset) - train_size
    
    lengths = [train_size, test_size]
    
    subsets = random_split(dataset, lengths, generator=generator)
    train_subset, test_subset = subsets[0], subsets[1]

    return train_subset, test_subset

def create_dataloaders(cfg):
    dataset_path = cfg.dataset.paths.local_path

    if cfg.dataset.name == 'AudioMNIST':
        test_set = MNISTDataset(
            path_to_data=dataset_path,
            split='test',
            target_sample_rate=cfg.dataset.sample_rate,
        )
        train_set = MNISTDataset(
            path_to_data=dataset_path,
            split='train',
            target_sample_rate=cfg.dataset.sample_rate,
        )
        train_set, dev_set = train_test_split(cfg, train_set, test_ratio=cfg.dataloader.dev_size)
    elif cfg.dataset.name == 'RESD':
        test_set = RESDDataset(
            split='test',
            target_sample_rate=cfg.dataset.sample_rate,
        )
        train_set = RESDDataset(
            split='train',
            target_sample_rate=cfg.dataset.sample_rate,
        )
        train_set, dev_set = train_test_split(cfg, train_set, test_ratio=cfg.dataloader.dev_size)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")
        
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_set, 
               batch_size=cfg.dataloader.batch_size, 
               shuffle=cfg.dataloader.shuffle,
               collate_fn=zero_pad_collate_fn,
               num_workers=cfg.dataloader.num_workers,
               generator=generator if cfg.dataloader.shuffle else None

               )
    
    dev_loader = DataLoader(dev_set, 
               batch_size=cfg.dataloader.batch_size, 
               shuffle=False,
               collate_fn=zero_pad_collate_fn,
               num_workers=cfg.dataloader.num_workers
               )
    
    test_loader = DataLoader(test_set, 
               batch_size=cfg.dataloader.batch_size, 
               shuffle=False,
               collate_fn=zero_pad_collate_fn,
               num_workers=cfg.dataloader.num_workers
               )
    
    return train_loader, dev_loader, test_loader
