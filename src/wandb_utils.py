from omegaconf import OmegaConf


def init_wandb(cfg, model=None):
    wandb_cfg = cfg.get("wandb")
    if wandb_cfg is None or not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled in config, but the wandb package is not installed.") from exc

    run = wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.get("entity") or None,
        name=wandb_cfg.get("run_name") or cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=wandb_cfg.get("mode", "online"),
        group=wandb_cfg.get("group") or None,
        job_type=wandb_cfg.get("job_type") or "train",
        tags=list(wandb_cfg.get("tags", [])),
        notes=wandb_cfg.get("notes") or None,
        save_code=bool(wandb_cfg.get("save_code", False)),
    )

    if model is not None and wandb_cfg.get("watch_model", False):
        run.watch(
            model,
            log=wandb_cfg.get("watch_log", "gradients"),
            log_freq=wandb_cfg.get("watch_log_freq", 100),
        )

    return run


def log_wandb_metrics(run, metrics, split=None, epoch=None):
    if run is None:
        return

    payload = {}
    for metric_name, value in metrics.items():
        key = f"{split}/{metric_name}" if split else metric_name
        payload[key] = value

    if epoch is not None:
        payload["epoch"] = epoch

    run.log(payload)


def finish_wandb(run):
    if run is not None:
        run.finish()
