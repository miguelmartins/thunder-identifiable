from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from omegaconf import DictConfig
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             jaccard_score)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..models.pretrained_models import load_pretrained_model
from ..utils.constants import UtilsConstants
from ..utils.data import PatchDataset, get_data
from ..utils.downstream_metrics import compute_metric, compute_metrics
from ..utils.pgd_attack_linear import PGDImageAttack
from ..utils.utils import log_metrics, save_outputs


def _pil_from_any(img):
    if isinstance(img, torch.Tensor):
        return T.ToPILImage()(img)
    return img


def adversarial_attack(
    cfg: DictConfig,
    dataset_name: str,
    model_name: str,
    image_pre_loading: bool,
    adaptation_type: str,
    device: torch.device,
    base_data_folder: str,
    wandb_base_folder: str,
    res_folder: str,
    model_cls: Callable = None,
):
    """Run PGD avdersarial attack.

    :param cfg: configuration file (hydra config).
    :param dataset_name: name of the dataset.
    :param model_name: name of the model.
    :param image_pre_loading: whether to preload images.
    :param adaptation_type: type of model adaptation.
    :param device: computation device.
    :param base_data_folder: base path to the data.
    :param wandb_base_folder: folder name for wandb logging.
    :param res_folder: path where to save outputs.
    :return: dictionary with adversarial attack metrics.
    """

    dataset_seed = (
        int(hashlib.sha256(dataset_name.encode()).hexdigest(), 16)
        + UtilsConstants.DEFAULT_SEED.value
    ) % (2**31)

    if adaptation_type == "lora":
        raise ValueError("LoRA not supported for adversarial pixel attack; use frozen.")

    data_paths = get_data(dataset_name, base_data_folder)
    img_paths = data_paths["test"]["images"]
    label_paths = data_paths["test"]["labels"]

    linear_ckpt_path = os.path.join(
        os.environ["THUNDER_BASE_DATA_FOLDER"],
        "outputs",
        "ckpts",
        dataset_name,
        model_name,
        adaptation_type,
        "best_model.pth",
    )

    if not os.path.exists(linear_ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint file not found at {linear_ckpt_path}. Please train a linear probe first."
        )

    ckpt = torch.load(linear_ckpt_path, weights_only=True)
    state_dict = ckpt["task_specific_model"]
    out_features, in_features = state_dict["linear.weight"].shape

    linear = nn.Linear(in_features, out_features)
    new_state_dict = {k.replace("linear.", ""): v for k, v in state_dict.items()}

    # Load into your nn.Linear layer
    linear = nn.Linear(in_features, out_features)
    linear.load_state_dict(new_state_dict)

    subset_indices = random.sample(
        range(len(img_paths)), k=min(cfg.task.nb_attack_images, len(img_paths))
    )

    if model_cls is not None:
        fm_model = model_cls
        preprocess = model_cls.get_transform()
        extract_embedding = model_cls.get_embeddings
        fm_model.to(device)
    else:
        fm_model, preprocess, extract_embedding = load_pretrained_model(
            cfg, adaptation_type, device
        )

    fm_model.eval()

    subset_dataset = Subset(
        PatchDataset(
            img_paths,
            label_paths,
            transform=preprocess,
            task_type="linear_probing",
            dataset_name=dataset_name,
            base_data_folder=base_data_folder,
            embeddings_folder=None,
            image_pre_loading=image_pre_loading,
            embedding_pre_loading=False,
        ),
        subset_indices,
    )

    gen = torch.Generator().manual_seed(dataset_seed)
    dl = DataLoader(
        subset_dataset,
        batch_size=cfg.task.attack_batch_size,
        shuffle=False,
        num_workers=cfg.adaptation.num_workers,
        generator=gen,
    )

    # attacker -----------------------------------------------------------
    atk_cfg = cfg.task.attack
    attacker = PGDImageAttack(
        fm_model,
        linear,
        extract_embedding,
        eps=float(atk_cfg.eps),
        alpha=float(atk_cfg.alpha),
        num_steps=int(atk_cfg.n_steps),
        norm=str(getattr(atk_cfg, "norm", "linf")),
        random_start=getattr(atk_cfg, "random_start", True),
        device=device,
    ).to(device)

    drift: Dict[str, List[float]] = defaultdict(list)

    all_clean_preds = []
    all_adv_preds = []
    all_labels = []

    for batch in tqdm(dl, desc="Adversarial PGD"):
        imgs = batch["image"] if isinstance(batch, dict) else batch
        imgs = imgs.to(device)

        labels = batch["label"] if isinstance(batch, dict) else batch
        labels = labels.to(device)

        clean_logits, adv_logits = attacker.perturb(imgs, labels)

        clean_preds = clean_logits.argmax(dim=1)
        adv_preds = adv_logits.argmax(dim=1)

        all_clean_preds.append(clean_preds.cpu())
        all_adv_preds.append(adv_preds.cpu())
        all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_clean_preds = torch.cat(all_clean_preds).cpu().numpy()
    all_adv_preds = torch.cat(all_adv_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    # Compute metrics before and after attack
    clean_metrics = compute_metrics(None, all_clean_preds, all_labels)
    adv_metrics = compute_metrics(None, all_adv_preds, all_labels)
    metrics = {
        "clean": clean_metrics,
        "adversarial": adv_metrics,
    }

    # Drop in metrics (mean and confidence intervals)
    f1_drop = compute_metric(
        all_labels,
        np.concatenate([all_clean_preds[:, None], all_adv_preds[:, None]], axis=1),
        lambda y, y_pred: f1_score(y_true=y, y_pred=y_pred[:, 0], average="macro")
        - f1_score(y_true=y, y_pred=y_pred[:, 1], average="macro"),
        label_indices=np.arange(len(all_labels)),
    )
    accuracy_drop = compute_metric(
        all_labels,
        np.concatenate([all_clean_preds[:, None], all_adv_preds[:, None]], axis=1),
        lambda y, y_pred: accuracy_score(y_true=y, y_pred=y_pred[:, 0])
        - accuracy_score(y_true=y, y_pred=y_pred[:, 1]),
        label_indices=np.arange(len(all_labels)),
    )
    jaccard_drop = compute_metric(
        all_labels,
        np.concatenate([all_clean_preds[:, None], all_adv_preds[:, None]], axis=1),
        lambda y, y_pred: jaccard_score(y_true=y, y_pred=y_pred[:, 0], average="macro")
        - jaccard_score(y_true=y, y_pred=y_pred[:, 1], average="macro"),
        label_indices=np.arange(len(all_labels)),
    )
    balanced_accuracy_drop = compute_metric(
        all_labels,
        np.concatenate([all_clean_preds[:, None], all_adv_preds[:, None]], axis=1),
        lambda y, y_pred: balanced_accuracy_score(y_true=y, y_pred=y_pred[:, 0])
        - balanced_accuracy_score(y_true=y, y_pred=y_pred[:, 1]),
        label_indices=np.arange(len(all_labels)),
    )

    metrics["drop"] = {
        "f1": f1_drop,
        "accuracy": accuracy_drop,
        "jaccard": jaccard_drop,
        "balanced_accuracy": balanced_accuracy_drop,
    }

    # save ---------------------------------------------------------------
    save_outputs(res_folder, metrics)

    # W&B ---------------------------------------------------------------
    for setting in ["clean", "adversarial"]:
        log_metrics(wandb_base_folder, metrics[setting], setting, step=0)

    return metrics
