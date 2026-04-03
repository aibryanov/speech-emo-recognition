import shutil
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import precision_recall_fscore_support

from src.wandb_utils import log_wandb_metrics


def _copmute_metrics(y_true, y_pred, loss, total_examples, average='macro'):
    prec, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )

    return {
        'precision': float(prec),
        'recall': float(recall),
        'f1': float(f1),
        'loss': float(loss / total_examples),
    }


def _get_checkpoint_dir(cfg):
    checkpoint_dir = Path(cfg.train.checkpoint.dir) / cfg.experiment_name

    if checkpoint_dir.is_absolute():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    try:
        from hydra.utils import get_original_cwd

        checkpoint_dir = Path(get_original_cwd()) / checkpoint_dir
    except Exception:
        checkpoint_dir = Path.cwd() / checkpoint_dir

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir


def _sort_saved_checkpoints(saved_checkpoints, mode='max'):
    reverse = mode == 'max'

    return sorted(saved_checkpoints, key=lambda checkpoint: checkpoint['score'], reverse=reverse)


def _load_saved_checkpoints(checkpoint_dir, monitor_name):
    saved_checkpoints = []

    for checkpoint_path in checkpoint_dir.glob(f'{monitor_name}_*'):
        if not checkpoint_path.is_dir():
            continue

        checkpoint_name = checkpoint_path.name.removeprefix(f'{monitor_name}_')
        checkpoint_score = checkpoint_name.split('_epoch_')[0]

        try:
            checkpoint_score = float(checkpoint_score)
        except ValueError:
            continue

        saved_checkpoints.append(
            {
                'path': checkpoint_path,
                'score': checkpoint_score,
            }
        )

    return saved_checkpoints


def _save_checkpoint(cfg, model, optimizer, checkpoint_dir, epoch, metrics, monitor_name):
    monitor_value = metrics[monitor_name]
    checkpoint_path = checkpoint_dir / f'{monitor_name}_{monitor_value:.4f}'

    if checkpoint_path.exists():
        checkpoint_path = checkpoint_dir / f'{monitor_name}_{monitor_value:.4f}_epoch_{epoch:03d}'

    checkpoint_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_path / 'model.pt')
    torch.save(optimizer.state_dict(), checkpoint_path / 'optimizer.pt')
    OmegaConf.save(cfg, checkpoint_path / 'config.yaml')

    return checkpoint_path


def _update_top_k_checkpoints(cfg, model, optimizer, saved_checkpoints, checkpoint_dir, metrics, epoch):
    top_k = cfg.train.checkpoint.top_k
    monitor_name = cfg.train.checkpoint.monitor
    mode = cfg.train.checkpoint.mode

    if top_k <= 0:
        return saved_checkpoints

    if mode not in ['max', 'min']:
        raise ValueError(f'Unsupported checkpoint mode: {mode}')

    if monitor_name not in metrics:
        raise ValueError(f'Metric {monitor_name} is not available in metrics.')

    checkpoint_path = _save_checkpoint(
        cfg,
        model,
        optimizer,
        checkpoint_dir,
        epoch,
        metrics,
        monitor_name,
    )

    saved_checkpoints.append(
        {
            'path': checkpoint_path,
            'score': metrics[monitor_name],
        }
    )
    saved_checkpoints = _sort_saved_checkpoints(saved_checkpoints, mode=mode)

    while len(saved_checkpoints) > top_k:
        removed_checkpoint = saved_checkpoints.pop(-1)
        shutil.rmtree(removed_checkpoint['path'], ignore_errors=True)

    return saved_checkpoints


def _append_split_history(metrics, history, split='train'):
    for metric in metrics.keys():
        history[f"{split}_{metric}"].append(metrics[f"{metric}"])

    return

def _print_metrics(metrics, epoch, num_epochs, split='TRAIN'):
    print(f"EPOCH [{epoch}/{num_epochs}] | {split} METRICS: loss={metrics['loss']:.4f} , precision={metrics['precision']:.4f} , recall={metrics['recall']:.4f} , f1={metrics['f1']:.4f}")

    return 

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
        raise ValueError("dev_loader is empty.")

    metrics = _copmute_metrics(
        all_labels,
        all_predictions,
        total_loss,
        total_examples,
        average=average,
    )

    return metrics

def train(cfg, model, train_loader, dev_loader=None, wandb_run=None):
    device = cfg.train.device
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
    checkpoint_enabled = cfg.train.checkpoint.get('enabled', False)
    checkpoint_dir = None
    saved_checkpoints = []

    if checkpoint_enabled:
        checkpoint_dir = _get_checkpoint_dir(cfg)
        saved_checkpoints = _load_saved_checkpoints(checkpoint_dir, cfg.train.checkpoint.monitor)
        saved_checkpoints = _sort_saved_checkpoints(saved_checkpoints, mode=cfg.train.checkpoint.mode)

        while len(saved_checkpoints) > cfg.train.checkpoint.top_k:
            removed_checkpoint = saved_checkpoints.pop(-1)
            shutil.rmtree(removed_checkpoint['path'], ignore_errors=True)

    history = {
        "train_loss": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "dev_loss": [],
        "dev_precision": [],
        "dev_recall": [],
        "dev_f1": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()

        running_loss = 0.0
        all_labels = []
        all_predictions = []
        total_examples = 0

        for step, (wavs, labels) in enumerate(train_loader, start=1):
            wavs = wavs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(wavs)    # (batch_size, classes_num)
            loss = loss_fn(logits, labels)
            predictions = logits.argmax(dim=1)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_examples += batch_size

            running_loss += loss.item() * batch_size
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

            if log_every and step % log_every == 0:
                if total_examples == 0:
                    raise ValueError("train_loader is empty.")

                metrics = _copmute_metrics(all_labels, all_predictions, running_loss, total_examples, average=metrics_average)

                _print_metrics(metrics, epoch, num_epochs)
                del metrics

        print(f"–––EPOCH {epoch} ENDED. EVALUATING–––")

        if dev_loader is not None:
            metrics = evaluate(model, dev_loader, loss_fn, device, average=metrics_average)
            _print_metrics(metrics, epoch, num_epochs, split='DEV')
            _append_split_history(metrics, history, split='dev')
            log_wandb_metrics(wandb_run, metrics, split="dev", epoch=epoch)

            if checkpoint_enabled:
                saved_checkpoints = _update_top_k_checkpoints(
                    cfg,
                    model,
                    optimizer,
                    saved_checkpoints,
                    checkpoint_dir,
                    metrics,
                    epoch,
                )

            del metrics

        if total_examples == 0:
            raise ValueError("train_loader is empty.")
        
        metrics = _copmute_metrics(all_labels, all_predictions, running_loss, total_examples, average=metrics_average)
        _print_metrics(metrics, epoch, num_epochs, split='TRAIN')
        _append_split_history(metrics, history, split='train')
        log_wandb_metrics(wandb_run, metrics, split="train", epoch=epoch)

        if checkpoint_enabled and dev_loader is None:
            saved_checkpoints = _update_top_k_checkpoints(
                cfg,
                model,
                optimizer,
                saved_checkpoints,
                checkpoint_dir,
                metrics,
                epoch,
            )

        del metrics

    return history

        
