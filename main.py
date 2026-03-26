import hydra
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.loader import create_dataloaders
from src.model import LSTMClassifier
from src.train import evaluate, train


def _format_metrics(prefix, metrics):
    return (
        f"{prefix} "
        f"loss={metrics['loss']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_loader, dev_loader, test_loader = create_dataloaders(cfg)
    model = LSTMClassifier(cfg)

    history = train(cfg, model, train_loader, dev_loader)

    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss()
    test_metrics = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        average=cfg.train.get("metrics_average", "macro"),
    )

    print(_format_metrics("test", test_metrics))

    return history


if __name__ == "__main__":
    main()
