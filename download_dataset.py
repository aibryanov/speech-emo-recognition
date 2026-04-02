import logging
from pathlib import Path

import hydra
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from omegaconf import DictConfig


logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)


def is_non_empty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    local_path = Path(cfg.dataset.paths.local_path)
    hf_path = cfg.dataset.paths.hf_path

    local_path.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(hf_path)
    except Exception:
        raise RuntimeError(f"Could not download from HF! Current HF path from config is {hf_path}")
    
if __name__ == "__main__":
    main()
