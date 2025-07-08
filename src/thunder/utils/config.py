import os
from typing import Optional

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


def get_config(
    task: Optional[str] = None,
    checkpoint: Optional[str] = None,
    dataset: Optional[str] = None,
    pretrained_model: Optional[str] = None,
    adaptation: Optional[str] = None,
    data_loading_type: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    embedding_recomputing: Optional[str] = None,
    model_retraining: Optional[str] = None,
    **kwargs,
) -> DictConfig:
    params = {
        "adaptation": adaptation,
        "ckpt_saving": checkpoint,
        "dataset": dataset,
        "data_loading": data_loading_type,
        "pretrained_model": pretrained_model,
        "task": task,
        "wandb": wandb_mode,
        "embedding_recomputing": embedding_recomputing,
        "model_retraining": model_retraining,
    }

    overrides = [f"+{k}={v}" for k, v in params.items() if v is not None]
    overrides += [f"++{k}={v}" for k, v in kwargs.items() if v is not None]
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides)

    return cfg
