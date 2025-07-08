import copy
import json
import logging
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from ..tasks.image_retrieval import topk_retrieval
from ..utils.data import get_data
from ..utils.downstream_metrics import compute_metrics
from ..utils.utils import log_metrics, save_outputs


def simple_shot(
    dataset_name: str,
    base_data_folder: str,
    train_embs: np.array,
    test_embs: np.array,
    test_labels: np.array,
    res_folder: str,
    wandb_base_folder: str,
) -> None:
    """
    Performing few-shot classification with SimpleShot (https://arxiv.org/abs/1911.04623).
    :param dataset_name: name of the dataset.
    :param base_data_folder: base folder storing data.
    :param train_embs: embeddings of training samples.
    :param test_embs: embeddings of test samples.
    :param test_labels: ground-truth test labels.
    :param res_folder: folder to save results.
    :param wandb_base_folder: w&b folder.
    """
    # Loading data
    data = get_data(dataset_name, base_data_folder)
    few_shot_data = data["train_few_shot"]
    nb_shots = few_shot_data.keys()
    simple_shot_metrics = {}
    for nb_shot in nb_shots:
        nb_shot_data = few_shot_data[nb_shot]
        nb_shot_images = nb_shot_data["images"]
        nb_shot_labels = nb_shot_data["labels"]
        assert len(nb_shot_images) == len(nb_shot_labels) == 1000
        out_pred = []
        logging.info(f"{nb_shot}-shot classification")
        for i in tqdm(range(len(nb_shot_images))):
            # Selecting few shot samples
            support_images = nb_shot_images[i]
            support_embs = train_embs[support_images]
            support_labels = nb_shot_labels[i]
            assert len(support_embs) == len(support_labels)
            # Centering support embeddings
            mean_support_emb = support_embs.mean(axis=0)
            assert mean_support_emb.shape == (support_embs.shape[1],)
            support_embs -= mean_support_emb

            # Simpleshot per-class mean
            label2embs = defaultdict(list)
            for j in range(len(support_labels)):
                label2embs[support_labels[j]].append(
                    np.expand_dims(support_embs[j], axis=0)
                )

            simple_shot_embs, simple_shot_labels = [], []
            for label in label2embs.keys():
                simple_shot_labels.append(label)
                mean_class_embs = np.concatenate(label2embs[label], axis=0).mean(axis=0)
                simple_shot_embs.append(np.expand_dims(mean_class_embs, axis=0))
            simple_shot_embs = np.concatenate(simple_shot_embs, axis=0)
            simple_shot_labels = np.array(simple_shot_labels)

            # Running knn
            metrics, *_ = topk_retrieval(
                simple_shot_embs,
                simple_shot_labels,
                test_embs - mean_support_emb,
                test_labels,
                [1],
                disable_progress_bar=True,
                compute_ci=False,
            )

            # Logging
            out_pred.append(metrics[1]["per_sample_pred"])

        # Few-shot-specific majority voting
        out_pred = np.array(out_pred)
        out_pred = np.array(
            [
                np.bincount(out_pred[:, i]).argmax().item()
                for i in range(out_pred.shape[1])
            ]
        )
        nb_shot_metrics = compute_metrics(None, out_pred, np.array(metrics[1]["label"]))

        # Logging
        simple_shot_metrics[nb_shot] = nb_shot_metrics
        log_metrics(wandb_base_folder, nb_shot_metrics, "test", step=int(nb_shot))

    # Logging
    save_outputs(res_folder, simple_shot_metrics)
