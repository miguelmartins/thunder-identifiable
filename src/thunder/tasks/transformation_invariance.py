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
import torch.nn.functional as F
import torchvision.transforms as T
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from ..models.adapters import get_model_lora_names, init_adapters
from ..models.pretrained_models import load_pretrained_model
from ..utils.constants import UtilsConstants
from ..utils.data import PatchDataset, get_data
from ..utils.transforms import get_invariance_transforms, set_transform_seed
from ..utils.utils import save_outputs, set_seed


def _pil_from_any(img):
    """
    Ensuring input is a PIL image.

    :param img: input image (tensor or PIL).
    :return: PIL image.
    """
    if isinstance(img, torch.Tensor):
        return T.ToPILImage()(img)
    return img


def _compute_batch_embeddings(
    preprocess: T.Compose,
    images_pil: List["PIL.Image.Image"],
    device: torch.device,
    extract_embedding,
    fm_model,
):
    """
    Preprocessing and computing embeddings for a batch of images.

    :param preprocess: preprocessing transformation.
    :param images_pil: list of PIL images.
    :param device: device to perform computations on.
    :param extract_embedding: function to extract embeddings.
    :param fm_model: model to use for embedding extraction.
    :return: batch embeddings tensor.
    """
    batch = torch.stack([preprocess(img) for img in images_pil]).to(device)
    with torch.no_grad():
        return extract_embedding(batch, fm_model, "linear_probing")


def transformation_invariance(
    cfg: DictConfig,
    dataset_name: str,
    model_name: str,
    image_pre_loading: bool,
    adaptation_type: str,
    device: str,
    base_data_folder: str,
    wandb_base_folder: str,
    res_folder: str,
) -> dict:
    """
    Computing transformation invariance metrics for a given model and dataset.

    :param cfg: configuration file (hydra config).
    :param dataset_name: name of the dataset.
    :param model_name: name of the model.
    :param image_pre_loading: whether to preload images.
    :param adaptation_type: type of model adaptation.
    :param device: computation device.
    :param base_data_folder: base path to the data.
    :param wandb_base_folder: folder name for wandb logging.
    :param res_folder: path where to save outputs.
    :return: dictionary with invariance metrics.
    """

    # We make sure that for different foundation models the same transformation params are randomly sampled for the same dataset
    dataset_seed = (
        int(hashlib.sha256(dataset_name.encode()).hexdigest(), 16)
        + UtilsConstants.DEFAULT_SEED.value
    ) % (2**31)

    if adaptation_type == "lora":
        raise ValueError(
            "Lora can not be chosen for this task, 'adaptation' should be set to 'frozen'"
        )
    # ---------------------------------------------------------------------
    # Loading training and validation datasets
    # ---------------------------------------------------------------------
    data_paths = get_data(dataset_name, base_data_folder)

    train_ds = PatchDataset(
        data_paths["train"]["images"],
        data_paths["train"]["labels"],
        transform=T.ToTensor(),  # transformations handled manually later
        task_type="linear_probing",
        dataset_name=dataset_name,
        base_data_folder=base_data_folder,
        embeddings_folder=None,
        image_pre_loading=image_pre_loading,
        embedding_pre_loading=False,
    )

    val_ds = PatchDataset(
        data_paths["val"]["images"],
        data_paths["val"]["labels"],
        transform=T.ToTensor(),  # transformations handled manually later
        task_type="linear_probing",
        dataset_name=dataset_name,
        base_data_folder=base_data_folder,
        embeddings_folder=None,
        image_pre_loading=image_pre_loading,
        embedding_pre_loading=False,
    )

    train_val_ds = ConcatDataset([train_ds, val_ds])

    # ---------------------------------------------------------------------
    # Selecting a random subset for evaluation
    # ---------------------------------------------------------------------
    subset_indices = random.sample(
        range(len(train_val_ds)),
        k=min(cfg.task.get("nb_images"), len(train_val_ds)),
    )

    subset_dataset = Subset(
        train_val_ds,
        subset_indices,
    )
    generator = torch.Generator()
    generator.manual_seed(dataset_seed)

    dataloader = DataLoader(
        subset_dataset,
        batch_size=cfg.task.transformation_invariance_batch_size,
        shuffle=False,
        num_workers=cfg.adaptation.num_workers,
        generator=generator,
    )

    # ---------------------------------------------------------------------
    # Loading pretrained foundation model (FM) and preprocessing
    # ---------------------------------------------------------------------
    fm_model, preprocess, extract_embedding = load_pretrained_model(
        cfg, adaptation_type, device
    )

    fm_model.eval()

    set_transform_seed(dataset_seed)

    invariance_transforms = get_invariance_transforms()

    # ---------------------------------------------------------------------
    # Main evaluation loop: compute similarities between original and transformed embeddings
    # ---------------------------------------------------------------------
    similarities: Dict[str, List[float]] = defaultdict(list)
    params_logged: Dict[str, List[dict]] = defaultdict(list)
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing invariance")):
        # Extracting images from batch
        batch_images = batch["image"] if isinstance(batch, dict) else batch

        # Convert batch to PIL images
        batch_pil = [_pil_from_any(img) for img in batch_images]

        # Compute embeddings for original images
        orig_emb = _compute_batch_embeddings(
            preprocess, batch_pil, device, extract_embedding, fm_model
        )
        # Loop over each transformation
        for name, tf in invariance_transforms.items():
            if name == "identity":
                continue

            # tf(img) now returns (aug, params)
            tf_results = [tf(img) for img in batch_pil]
            tf_pil = [aug for aug, _ in tf_results]
            tf_params = [p for _, p in tf_results]
            # embeddings & similarity
            tf_emb = _compute_batch_embeddings(
                preprocess, tf_pil, device, extract_embedding, fm_model
            )
            cos_sim = F.cosine_similarity(orig_emb, tf_emb, dim=1)

            similarities[name].extend(cos_sim.cpu().numpy().tolist())
            params_logged[name].extend(tf_params)

    # ---------------------------------------------------------------------
    # Aggregating results and saving
    # ---------------------------------------------------------------------
    invariance_metrics = {
        name: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": [float(v) for v in vals],  # saving all individual values
            "params": params_logged[
                name
            ],  # saving all individual transformation params
        }
        for name, vals in similarities.items()
    }

    # Saving metrics to JSON
    save_outputs(res_folder, invariance_metrics)

    # Logging to WandB
    metric_prefix = f"{wandb_base_folder}/cosine_similarity"
    wandb.log(
        {
            f"{metric_prefix}/{name}": stats["mean"]
            for name, stats in invariance_metrics.items()
        }
    )
