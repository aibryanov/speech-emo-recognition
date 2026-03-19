import hydra
from omegaconf import DictConfig
from src.dataset import audioMNIST
from src.hf_dataset import hf_audioMNIST



@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(config: DictConfig):
    
    
    print(type(hf_audioMNIST))
    print(type(audioMNIST))

if __name__ == '__main__':
    main()