import json
import numpy as np
from omegaconf import DictConfig
import os
import random
import torch
import wandb
import contextlib
import h5py
from typing import Dict, Any, Optional


def get_hyperaparams_dict(cfg: DictConfig) -> dict:
    """
    Creating dict of hyperparameter sweeps.
    :param cfg: config defining the job to run.
    """
    hyperparams_dict = []
    for lr in cfg.adaptation.lr:
        for weight_decay in cfg.adaptation.weight_decay:
            hyperparams_dict.append(
                {
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )
    return hyperparams_dict


def log_loss(
    wandb_base_folder: str,
    lr: float,
    weight_decay: float,
    losses: list,
    split: str,
    epoch: int,
):
    """
    Logging loss.
    :param wandb_base_folder: w&b folder.
    :param lr: learning rate.
    :param weight_decay: weight decay.
    :param losses: list of losses.
    :param split: data split.
    :param epoch: training epoch.
    """
    logs = {
        f"{wandb_base_folder}/{split}_lr_{lr}_weight_decay_loss_"
        f"{weight_decay}": np.array(losses).mean().item()
    }
    wandb.log(logs, step=epoch)


def log_metrics(wandb_base_folder: str, metrics: dict, split: str, step: int):
    """
    Logging dict of metrics.
    :param wandb_base_folder: w&b folder.
    :param metrics: dict of metrics.
    :param split: data split.
    :param step: w&b log step.
    """
    logs = {}
    for metric in metrics.keys():
        if (
            metric != "label"
            and "per_sample" not in metric
            and metrics[metric] is not None
        ):
            logs[f"{wandb_base_folder}/{split}_{metric}"] = metrics[metric][
                "metric_score"
            ]
    wandb.log(logs, step=step)


@contextlib.contextmanager
def local_seed(seed: int):
    """
    Fix a local random seed in a context and restore the previous states.
    :param seed: an integer seed value.
    """
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_state = torch.cuda.get_rng_state()
    set_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_state)


def save_outputs(res_folder: str, outputs: dict) -> None:
    """
    Saving run outputs.
    :param res_folder: path where to save outputs.
    :param outputs: dictionary of outputs.
    """
    with open(os.path.join(res_folder, "outputs.json"), "w") as outfile:
        json.dump(outputs, outfile)


def set_seed(seed: int) -> None:
    """
    Setting all seeds to make results reproducible.
    :param seed: an integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def wb_mask(
    bg_img: torch.Tensor,
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    dataset_classes: list,
) -> wandb.Image:
    """
    Generating semantic masks visualization with wandb.
    :param bg_img: background image.
    :param pred_mask: mask prediction.
    :param true_mask: ground-truth mask.
    :param dataset_classes: list of class names.
    Adapted from https://colab.research.google.com/drive/1SOVl3EvW82Q4QKJXX6JtHye4wFix_P4J#scrollTo=iCXdt_Fq287_
    """
    labels = {}
    for i, label in enumerate(dataset_classes):
        labels[i] = label

    mask_list = []
    for j in range(bg_img.shape[0]):
        mask = wandb.Image(
            bg_img[j],
            masks={
                "prediction": {"mask_data": pred_mask[j], "class_labels": labels},
                "ground truth": {"mask_data": true_mask[j], "class_labels": labels},
            },
        )
        mask_list.append(mask)
    return mask_list
