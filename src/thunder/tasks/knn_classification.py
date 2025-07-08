import json
import logging
import os

import numpy as np

from ..tasks.image_retrieval import topk_retrieval
from ..utils.utils import log_metrics, save_outputs


def knn(
    embs: dict,
    labels: dict,
    k_vals: list,
    res_folder: str,
    wandb_base_folder: str,
) -> None:
    """
    Performing knn classification.
    :param embs: dict of embeddings.
    :param labels: dict of labels.
    :param k_vals: values of k to consider.
    :param res_folder: folder to save results.
    :param wandb_base_folder: w&b folder.
    """
    # Hyperparameter (k) search
    logging.info("Hyperparameter (k) search")
    val_metrics, *_ = topk_retrieval(
        embs["train"],
        labels["train"],
        embs["val"],
        labels["val"],
        k_vals,
        compute_ci=False,
    )

    # Logging
    for k in k_vals:
        log_metrics(wandb_base_folder, val_metrics[k], "val", step=k)

    # Picking best k
    best_k = None
    best_val_f1 = -float("inf")
    for k in k_vals:
        val_f1 = np.array(val_metrics[k]["f1"]["metric_score"]).mean().item()
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_k = k

    # knn test
    logging.info("Test")
    test_metrics, *_ = topk_retrieval(
        embs["train"],
        labels["train"],
        embs["test"],
        labels["test"],
        [best_k],
    )

    # Logging
    log_metrics(wandb_base_folder, test_metrics[best_k], "test", step=best_k)
    save_outputs(res_folder, test_metrics)
