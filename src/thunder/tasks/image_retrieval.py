import json
import logging
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb
from PIL import Image
from tqdm import tqdm

from ..utils.data import get_data
from ..utils.downstream_metrics import compute_metrics
from ..utils.utils import log_metrics, save_outputs


def normalize(embs: np.array) -> np.array:
    """
    Diviving each row embedding by its L2 norm.

    :pram embs: matrix where each row is an image embedding.
    :return row-normalized matrix.
    """

    embs_norm = np.linalg.norm(embs, axis=1, keepdims=True)
    normalized_embs = embs / embs_norm
    return normalized_embs


def topk_retrieval(
    keys: np.array,
    key_labels: np.array,
    queries: np.array,
    query_labels: np.array,
    k_vals: list,
    chunk_size: int = 10000,
    return_viz_data: bool = False,
    disable_progress_bar: bool = False,
    compute_ci: bool = True,
) -> tuple[dict, dict, dict]:
    """
    Computing similarities between queries and keys.
    :param keys: key embeddings.
    :param key_labels: labels of key images.
    :param queries: query embeddings.
    :param query_labels: labels of query images.
    :param k_vals: values of k to consider.
    :param chunk_size: maximum number of query embeddings for which we compute dot product similarity with all key embeddings.
    :param return_viz_data: whether to return data to visualize topk samples.
    :param disable_progress_bar: whether to hide the progress bar.
    :param compute_ci: whether to compute confidence intervals.
    :return dict of metrics, sorted images ids and viz data if required.
    """
    # Normalizing embeddings
    keys = normalize(keys)
    queries = normalize(queries)
    assert np.allclose(
        np.linalg.norm(keys, axis=1, keepdims=True), np.ones((len(keys), 1))
    ), "keys matrix is not row-normalized."
    assert np.allclose(
        np.linalg.norm(queries, axis=1, keepdims=True), np.ones((len(queries), 1))
    ), "queries matrix is not row-normalized."

    preds_per_k = defaultdict(list)
    sorted_ids_per_k = defaultdict(list)
    viz_data = []
    for row in tqdm(
        range(0, queries.shape[0], chunk_size), disable=disable_progress_bar
    ):
        # Dot product
        row_end = row + chunk_size
        dot_product = np.dot(queries[row:row_end], keys.transpose())

        for i in range(dot_product.shape[0]):
            # Sorting each row according to similarity scores
            row_argsort = dot_product[i].argsort()[::-1]
            row_key_labels = key_labels[row_argsort]

            for k in k_vals:
                k_key_labels = row_key_labels[:k]
                # Storing sorted ids
                sorted_ids_per_k[k].append(row_argsort[:k].copy().tolist())
                # Majority Voting prediction
                mv_pred = np.bincount(k_key_labels).argmax()
                preds_per_k[k].append(mv_pred.item())

            if return_viz_data and row == 0 and i < 8:
                viz_data.append(
                    {
                        "query_path_idx": row + i,
                        "key_path_idxs": row_argsort[:10],
                        "key_scores": dot_product[i][row_argsort][:10],
                    }
                )

    # Metrics
    metrics_per_k = {}
    for k in k_vals:
        metrics_per_k[k] = compute_metrics(
            None, np.array(preds_per_k[k]), query_labels, compute_ci=compute_ci
        )

    return metrics_per_k, sorted_ids_per_k, viz_data


def image_retrieval(
    train_embs: np.array,
    train_labels: np.array,
    test_embs: np.array,
    test_labels: np.array,
    base_data_folder: str,
    dataset_name: str,
    k_vals: list,
    res_folder: str = "",
    wandb_base_folder: str = "",
    chunk_size: int = 10000,
) -> dict:
    """
    Performing image retrieval where each test embedding is used as a
    query and each train embedding as a key. Reporting Acc@K and MVAcc@K (K=1,3,5,10).

    :param train_embs: embeddings extracted from the training images with the pretrained model.
    :param train_labels: labels of training images.
    :param test_embs: embeddings extracted from the test images with the pretrained model.
    :param test_labels: labels of test images.
    :param base_data_folder: path to the base folder storing data.
    :param dataset_name: name of the loaded dataset.
    :param k_vals: values of k to consider.
    :param res_folder: folder to save results.
    :param wandb_base_folder: w&b folder.
    :param chunk_size: maximum number of test embeddings for which we compute
                       dot product similarity with all train embeddings.
    :return metrics dict.
    """

    metrics, _, viz_data = topk_retrieval(
        train_embs,
        train_labels,
        test_embs,
        test_labels,
        k_vals,
        chunk_size,
        return_viz_data=True,
    )

    # Loading data
    data = get_data(dataset_name, base_data_folder)

    if dataset_name != "patch_camelyon":
        # Figure
        fig, axs = plt.subplots(8, 11, figsize=(30, 20))

        for i, sample in enumerate(viz_data):
            query_path = data["test"]["images"][sample["query_path_idx"]]

            query_im = Image.open(
                os.path.join(base_data_folder, dataset_name, query_path)
            ).convert("RGB")
            axs[i, 0].imshow(query_im)
            axs[i, 0].grid(False)
            axs[i, 0].axis("off")
            axs[i, 0].set_title("Query", fontsize=20)

            key_idxs = sample["key_path_idxs"]
            scores = sample["key_scores"]
            for j, key_idx in enumerate(key_idxs):
                key_path = data["train"]["images"][key_idx]
                key_im = Image.open(
                    os.path.join(base_data_folder, dataset_name, key_path)
                ).convert("RGB")
                axs[i, j + 1].imshow(key_im)
                axs[i, j + 1].grid(False)
                axs[i, j + 1].axis("off")
                axs[i, j + 1].set_title(round(scores[j], 3), fontsize=20)

        # Logging
        wandb.log({f"{wandb_base_folder}/image_retrieval_samples": fig})
        fig.savefig(
            os.path.join(res_folder, "retrieval_samples.pdf"),
            bbox_inches="tight",
            pad_inches=0,
        )
    else:
        logging.info("Retrieval samples are not visualized for path_camelyon.")

    # Logging
    for k in k_vals:
        log_metrics(wandb_base_folder, metrics[k], "test", step=k)
    save_outputs(res_folder, metrics)

    return metrics
