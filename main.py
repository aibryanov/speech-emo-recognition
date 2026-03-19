import hydra
from omegaconf import DictConfig



@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(config: DictConfig):
    
    
    return

if __name__ == '__main__':
    main()