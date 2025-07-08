import os
from collections.abc import Callable

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.models.vit.modeling_vit import ViTModel

from ..models.pretrained_models import load_pretrained_model
from ..utils.data import PatchDataset, get_data


def pre_computing_patch_embeddings(
    cfg: DictConfig,
    embeddings_folder: str,
    device: str,
    dataset_name: str,
    base_data_folder: str,
    data_compatible_tasks: list,
    adaptation_type: str,
    base_embeddings_folder: str,
    model_name: str,
    image_pre_loading: str,
    embedding_pre_loading: str,
    model_cls: Callable = None,
) -> None:
    """
    Pre-computing embeddings for each patch in a given dataset with a given pre-trained model.

    :param cfg: config defining the job to run.
    :param embeddings_folder: folder where to store computed embeddings.
    :param device: device to use (cpu, cuda).
    :param dataset_name: name of the dataset.
    :param base_data_folder: base folder storing data.
    :param data_compatible_tasks: list of tasks compatible with the dataset.
    :param adaptation_type: type of adaptation to use (frozen, lora).
    :param base_embeddings_folder: base folder storing embeddings.
    :param model_name: name of the pretrained model.
    :param image_pre_loading: whether to pre-load images in dataloader.
    :param embedding_pre_loading: whether to pre-load embeddings in dataloader.
    """
    # Loading data
    data = get_data(dataset_name, base_data_folder)

    # Loading pretrained model
    if model_cls is not None:
        pretrained_model = model_cls
        transform = model_cls.get_transform()
        extract_embedding = model_cls.get_embeddings
        pretrained_model.to(device)
    else:
        pretrained_model, transform, extract_embedding = load_pretrained_model(
            cfg, adaptation_type, device
        )
    pretrained_model.eval()

    # Pre-computing (and saving) embeddings
    if "linear_probing" in data_compatible_tasks:
        dataset_task_type = "linear_probing"
    else:
        dataset_task_type = "segmentation"
    for split in ["train", "val", "test"]:
        split_dataset = PatchDataset(
            data[split]["images"],
            data[split]["labels"],
            transform,
            dataset_task_type,
            dataset_name,
            base_data_folder,
            os.path.join(base_embeddings_folder, dataset_name, model_name, split),
            image_pre_loading,
            embedding_pre_loading,
        )
        split_dataloader = DataLoader(
            split_dataset,
            batch_size=cfg.task.pre_comp_emb_batch_size,
            shuffle=False,
            num_workers=1,
        )
        pre_computing_patch_embeddings_split(
            os.path.join(embeddings_folder, split),
            split_dataloader,
            pretrained_model,
            extract_embedding,
            dataset_task_type,
            device,
        )


@torch.inference_mode()
def pre_computing_patch_embeddings_split(
    embeddings_folder: str,
    dataloader: DataLoader,
    pretrained_model: ViTModel,
    extract_embedding: Callable[[torch.Tensor, ViTModel, str], torch.Tensor],
    task_type: str,
    device: torch.device | str = "cuda",
) -> None:
    """
    Pre-computing embeddings for each patch in a split of a given dataset with a given pre-trained model.

    :param embeddings_folder: folder where to store computed embeddings.
    :param dataloader: dataloader to load data batches.
    :param pretrained_model: pretrained model to use.
    :param extract_embedding: function to extract an embedding vector from an input image with the pretrained model.
    :param task_type: type of task to extract emebddings for (classification, segmentation).
    :param device: device to use (cpu, cuda).
    """
    os.makedirs(embeddings_folder, exist_ok=True)
    emb_path = os.path.join(embeddings_folder, "embeddings.h5")
    label_path = os.path.join(embeddings_folder, "labels.h5")

    # Open / create the files once.
    with h5py.File(emb_path, "a", libver="latest") as emb_h5, h5py.File(
        label_path, "a", libver="latest"
    ) as lab_h5:
        next_idx = max((int(k) for k in emb_h5.keys()), default=-1) + 1

        for batch in tqdm(dataloader, desc="Extracting patch embeddings"):
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            embeds = extract_embedding(imgs, pretrained_model, task_type=task_type)
            embeds = embeds.cpu().numpy().astype(np.float32, copy=False)
            labels = labels.cpu().numpy().astype(np.int64, copy=False)

            bs = embeds.shape[0]
            for i in range(bs):
                key = f"{next_idx + i}"

                emb_h5.create_dataset(
                    key,
                    data=embeds[i],
                    dtype="float32",
                )
                lab_h5.create_dataset(
                    key,
                    data=labels[i],
                    dtype="int64",
                )

            next_idx += bs
