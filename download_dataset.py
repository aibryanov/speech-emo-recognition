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


@hydra.main(version_base='1.3', config_path='configs')
def main(cfg: DictConfig):
    local_path = Path(cfg.dataset.paths.local_path)
    hf_path = cfg.dataset.paths.hf_path
    dataset_name = cfg.dataset.name.lower()

    local_path.mkdir(parents=True, exist_ok=True)

    # !FOR Dusha Dataset ONLY! #
    if dataset_name == "dusha":
        if is_non_empty_dir(local_path):
            print(f"{local_path} dataset files already exist.")
            return

        try:
            snapshot_download(
                repo_id=hf_path,
                repo_type="dataset",
                local_dir=local_path,
            )
        except Exception:
            raise RuntimeError(f"Could not download legacy HF dataset repo! Current HF path from config is {hf_path}")

        print(f"{local_path} dataset repo downloaded.")
        return

    # !FOR Datasets which supports load_dataset! #
    if local_path.exists():
        try: 
            load_from_disk(str(local_path))
            print(f"{str(local_path)} dataset already exists.")
            return
        except Exception:
            print(f"{str(local_path)} path already exists, but invalid HF dataset or path is empty! Trying to download...")

    try:
        dataset = load_dataset(hf_path)
    except Exception:
        raise RuntimeError(f"Could not download from HF! Current HF path from config is {hf_path}")
    
    try:
        dataset.save_to_disk(str(local_path))
    except Exception:
        raise RuntimeError(f"Could not save to disk! Current dataset local path is {local_path}")

if __name__ == "__main__":
    main()
