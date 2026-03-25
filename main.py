import hydra
from omegaconf import DictConfig
from src.dataset import MNISTDataset



@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(config: DictConfig):
    dataset = MNISTDataset(split='test')
    print(len(dataset))

if __name__ == '__main__':
    main()