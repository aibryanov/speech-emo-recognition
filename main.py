import hydra
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.loader import create_dataloaders
from src.model import LSTMClassifier
from src.train import evaluate, train
from src.wandb_utils import finish_wandb, init_wandb, log_wandb_metrics

def print_metrics(test_metrics):
    print("–––––TEST METRICS–––––")
    print(f"F1: {test_metrics['f1']}")
    print(f"Precision: {test_metrics['precision']}")
    print(f"Recall: {test_metrics['recall']}")
    
    
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    wandb_run = None

    try:
        train_loader, dev_loader, test_loader = create_dataloaders(cfg)
        model = LSTMClassifier(cfg)
        wandb_run = init_wandb(cfg, model)

        history = train(cfg, model, train_loader, dev_loader, wandb_run=wandb_run)

        device = cfg.train.device
        loss_fn = nn.CrossEntropyLoss()
        test_metrics = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            average=cfg.train.get("metrics_average", "macro"),
        )
        log_wandb_metrics(wandb_run, test_metrics, split="test", epoch=cfg.train.epochs)

        print_metrics(test_metrics)

        return history
    finally:
        finish_wandb(wandb_run)



if __name__ == "__main__":
    main()
