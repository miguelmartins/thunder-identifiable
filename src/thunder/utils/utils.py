import contextlib
import json
import logging
import os
import random
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf


def print_task_hyperparams(cfg: DictConfig, custom_name: Optional[str] = None) -> None:
    """
    Print dataset, model, and only the task-specific hyper-parameters
    from a full Hydra cfg, using classic ANSI colors.

    If `custom_name` is provided, it will be used instead of
    cfg.pretrained_model.model_name.
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BLUE = "\033[34m"
    WHITE = "\033[37m"
    GREEN = "\033[32m"
    RED = "\033[31m"

    task = cfg.task.type
    dataset_name = cfg.dataset.dataset_name

    # Safe-fetch the model name, falling back to custom_name if given
    if custom_name is not None:
        model_label = custom_name
    else:
        model_label = OmegaConf.select(
            cfg, "pretrained_model.model_name", default="<unknown model>"
        )

    # Choose where hyperparams live
    task_cfg = (
        cfg.task if task not in ["linear_probing", "segmentation"] else cfg.adaptation
    )

    sep = "-" * 60

    logging.info(f"\n{BOLD}{BLUE}\U0001f680  Experiment Info{RESET}")
    print(sep)
    print(f"{BLUE}Task   :{RESET} {WHITE}{task}{RESET}")
    print(f"{BLUE}Dataset:{RESET} {WHITE}{dataset_name}{RESET}")
    print(f"{BLUE}Model  :{RESET} {WHITE}{model_label}{RESET}")
    print(sep)

    print(f"\n{BOLD}{BLUE}Hyper-parameters{RESET}\n{sep}")

    # Fields to skip
    skip = {"compatible_adaptation_types", "base_embeddings_folder"}
    if task not in ["linear_probing", "segmentation"]:
        skip.add("type")

    # Print each hyperparam
    for key, val in task_cfg.items():
        if key in skip:
            continue

        # Special PGD block header
        if task == "adversarial_attack" and key == "attack":
            print(f"\n{RED}{BOLD}PGD attack hyper-parameters{RESET}")

        # Nested dicts
        if isinstance(val, (DictConfig, dict)):
            print(f"{BLUE}{BOLD}{key}{RESET}:")
            for subkey, subval in val.items():
                print(f"  {WHITE}{subkey}{RESET}: {subval}")
        else:
            print(f"{BLUE}{key}{RESET}: {WHITE}{val}{RESET}")

    print(sep)


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
        f"{wandb_base_folder}/{split}_lr_{lr}_weight_decay_{weight_decay}_loss": np.array(
            losses
        )
        .mean()
        .item()
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
    output_filename = os.path.join(res_folder, "outputs.json")
    with open(output_filename, "w") as outfile:
        json.dump(outputs, outfile)
    logging.info(f"Outputs saved at {output_filename}")


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
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


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
