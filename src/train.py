import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _compute_classification_metrics(labels, predictions, average="macro"):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=average,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _get_checkpoint_dir(cfg):
    checkpoint_dir = Path(cfg.train.checkpoint.get("dir", "checkpoints"))

    if checkpoint_dir.is_absolute():
        return checkpoint_dir

    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().runtime.output_dir) / checkpoint_dir
    except Exception:
        return checkpoint_dir


def _sort_checkpoints(checkpoints, mode):
    reverse = mode == "max"

    return sorted(checkpoints, key=lambda checkpoint: checkpoint["score"], reverse=reverse)


def _write_leaderboard(checkpoint_dir, checkpoints):
    leaderboard_path = checkpoint_dir / "leaderboard.json"
    payload = []

    for rank, checkpoint in enumerate(checkpoints, start=1):
        payload.append(
            {
                "rank": rank,
                "epoch": checkpoint["epoch"],
                "score": checkpoint["score"],
                "monitor": checkpoint["monitor"],
                "path": str(checkpoint["path"]),
                "metrics": checkpoint["metrics"],
            }
        )

    with leaderboard_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_checkpoint(cfg, checkpoint_dir, epoch, model, optimizer, metrics, monitor_name):
    monitor_value = metrics[monitor_name]
    checkpoint_name = f"epoch_{epoch:03d}_{monitor_name}_{monitor_value:.4f}"
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "monitor": monitor_name,
        },
        checkpoint_path / "checkpoint.pt",
    )
    OmegaConf.save(cfg, checkpoint_path / "config.yaml")

    with (checkpoint_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "monitor": monitor_name,
                "score": monitor_value,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    return checkpoint_path


def _update_top_checkpoints(
    cfg,
    checkpoint_dir,
    saved_checkpoints,
    epoch,
    model,
    optimizer,
    metrics,
):
    checkpoint_cfg = cfg.train.checkpoint
    top_k = checkpoint_cfg.top_k
    monitor_name = checkpoint_cfg.monitor
    mode = checkpoint_cfg.mode

    if monitor_name not in metrics:
        raise ValueError(f"Checkpoint monitor '{monitor_name}' is not available in metrics: {list(metrics.keys())}")

    checkpoint_path = _save_checkpoint(
        cfg,
        checkpoint_dir,
        epoch,
        model,
        optimizer,
        metrics,
        monitor_name,
    )

    saved_checkpoints.append(
        {
            "path": checkpoint_path,
            "epoch": epoch,
            "score": metrics[monitor_name],
            "monitor": monitor_name,
            "metrics": metrics,
        }
    )
    saved_checkpoints = _sort_checkpoints(saved_checkpoints, mode)

    while len(saved_checkpoints) > top_k:
        removed_checkpoint = saved_checkpoints.pop(-1)
        shutil.rmtree(removed_checkpoint["path"], ignore_errors=True)

    _write_leaderboard(checkpoint_dir, saved_checkpoints)

    return saved_checkpoints


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device, average="macro"):
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_labels = []
    all_predictions = []

    for wavs, labels in data_loader:
        wavs = wavs.to(device)
        labels = labels.to(device)

        logits = model(wavs)
        loss = loss_fn(logits, labels)

        predictions = logits.argmax(dim=1)
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    if total_examples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    metrics = _compute_classification_metrics(
        all_labels,
        all_predictions,
        average=average,
    )
    metrics["loss"] = float(total_loss / total_examples)

    return metrics


def train(cfg, model, train_loader, dev_loader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.get("weight_decay", 0.01),
    )

    num_epochs = cfg.train.epochs
    log_every = cfg.train.get("log_every", 0)
    metrics_average = cfg.train.get("metrics_average", "macro")
    checkpoint_cfg = cfg.train.get("checkpoint", {})
    checkpoint_enabled = checkpoint_cfg.get("enabled", False)
    checkpoint_dir = None
    saved_checkpoints = []

    if checkpoint_enabled:
        if dev_loader is None:
            raise ValueError("DEV checkpointing is enabled, but dev_loader is None.")

        checkpoint_dir = _get_checkpoint_dir(cfg)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "dev_loss": [],
        "dev_accuracy": [],
        "dev_precision": [],
        "dev_recall": [],
        "dev_f1": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()

        running_loss = 0.0
        total_examples = 0
        all_labels = []
        all_predictions = []

        for step, (wavs, labels) in enumerate(train_loader, start=1):
            wavs = wavs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(wavs)
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)

            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            running_loss += loss.item() * batch_size
            total_examples += batch_size
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

            if log_every and step % log_every == 0:
                train_loss = running_loss / total_examples
                step_metrics = _compute_classification_metrics(
                    all_labels,
                    all_predictions,
                    average=metrics_average,
                )
                print(
                    f"epoch {epoch}/{num_epochs} "
                    f"step {step}/{len(train_loader)} "
                    f"train_loss={train_loss:.4f} "
                    f"train_accuracy={step_metrics['accuracy']:.4f} "
                    f"train_precision={step_metrics['precision']:.4f} "
                    f"train_recall={step_metrics['recall']:.4f} "
                    f"train_f1={step_metrics['f1']:.4f}"
                )

        if total_examples == 0:
            raise ValueError("train_loader is empty.")

        epoch_train_loss = running_loss / total_examples
        epoch_train_metrics = _compute_classification_metrics(
            all_labels,
            all_predictions,
            average=metrics_average,
        )
        history["train_loss"].append(epoch_train_loss)
        history["train_accuracy"].append(epoch_train_metrics["accuracy"])
        history["train_precision"].append(epoch_train_metrics["precision"])
        history["train_recall"].append(epoch_train_metrics["recall"])
        history["train_f1"].append(epoch_train_metrics["f1"])

        metrics_message = (
            f"epoch {epoch}/{num_epochs} "
            f"train_loss={epoch_train_loss:.4f} "
            f"train_accuracy={epoch_train_metrics['accuracy']:.4f} "
            f"train_precision={epoch_train_metrics['precision']:.4f} "
            f"train_recall={epoch_train_metrics['recall']:.4f} "
            f"train_f1={epoch_train_metrics['f1']:.4f}"
        )

        if dev_loader is not None:
            dev_metrics = evaluate(
                model,
                dev_loader,
                loss_fn,
                device,
                average=metrics_average,
            )
            history["dev_loss"].append(dev_metrics["loss"])
            history["dev_accuracy"].append(dev_metrics["accuracy"])
            history["dev_precision"].append(dev_metrics["precision"])
            history["dev_recall"].append(dev_metrics["recall"])
            history["dev_f1"].append(dev_metrics["f1"])

            if checkpoint_enabled:
                saved_checkpoints = _update_top_checkpoints(
                    cfg,
                    checkpoint_dir,
                    saved_checkpoints,
                    epoch,
                    model,
                    optimizer,
                    dev_metrics,
                )

            metrics_message += (
                f" dev_loss={dev_metrics['loss']:.4f} "
                f"dev_accuracy={dev_metrics['accuracy']:.4f} "
                f"dev_precision={dev_metrics['precision']:.4f} "
                f"dev_recall={dev_metrics['recall']:.4f} "
                f"dev_f1={dev_metrics['f1']:.4f}"
            )

        print(metrics_message)

    return history
