import torch
import torch.nn as nn
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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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
    metrics["loss"] = total_loss / total_examples

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
            metrics_message += (
                f" dev_loss={dev_metrics['loss']:.4f} "
                f"dev_accuracy={dev_metrics['accuracy']:.4f} "
                f"dev_precision={dev_metrics['precision']:.4f} "
                f"dev_recall={dev_metrics['recall']:.4f} "
                f"dev_f1={dev_metrics['f1']:.4f}"
            )

        print(metrics_message)

    return history
