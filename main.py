import hydra
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.loader import create_dataloaders
from src.model import LSTMClassifier
from src.train import (
    build_classification_details,
    evaluate,
    print_classification_details,
    train,
    _get_dataset_class_names,
)
from src.wandb_utils import finish_wandb, init_wandb, log_wandb_confusion_matrix, log_wandb_metrics

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
        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.train.get("label_smoothing", 0.0))
        test_metrics, y_true, y_pred = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            average=cfg.train.get("metrics_average", "macro"),
            return_predictions=True,
        )
        log_wandb_metrics(wandb_run, test_metrics, split="test", epoch=cfg.train.epochs)

        print_metrics(test_metrics)

        class_names = _get_dataset_class_names(test_loader.dataset)
        details = build_classification_details(y_true, y_pred, class_names)
        if cfg.evaluation.get("print_per_class_metrics", True) or cfg.evaluation.get("print_confusion_matrix", True):
            print_classification_details(details, split="TEST")
        log_wandb_confusion_matrix(wandb_run, y_true, y_pred, class_names, split="test")

        return history
    finally:
        finish_wandb(wandb_run)



if __name__ == "__main__":
    main()
