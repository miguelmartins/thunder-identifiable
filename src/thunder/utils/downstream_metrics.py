import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    jaccard_score,
    roc_auc_score,
)
import torch
from typing import Union


def compute_metrics(
    out_proba: Union[torch.Tensor, np.array],
    out_pred: Union[torch.Tensor, np.array],
    label: Union[torch.Tensor, np.array],
    is_segmentation: bool = False
) -> dict:
    """
    Computing performance metrics.
    :param out_proba: tensor of predicted probabilities.
    :param out_pred: tensor of predicted labels.
    :param label: tensor of ground-truth labels.
    :return: dict of metrics.
    """
    if out_proba is not None:
        assert out_pred is None
        # Computing argmax
        out_pred = out_proba.argmax(dim=1)

        # torch -> numpy
        out_proba = out_proba.numpy()
        label = label.numpy()
        out_pred = out_pred.numpy()

    # Computing metrics
    accuracy = accuracy_score(y_true=label, y_pred=out_pred)
    f1 = f1_score(y_true=label, y_pred=out_pred, average="macro")
    jaccard = jaccard_score(y_true=label, y_pred=out_pred, average="macro")

    metrics = {"f1": f1,
               "accuracy": accuracy,
               "jaccard": jaccard}
    
    if not is_segmentation:
        balanced_accuracy = balanced_accuracy_score(y_true=label, y_pred=out_pred)
        if out_proba is not None:
            if out_proba.shape[1] > 2:
                roc_auc = roc_auc_score(
                    y_true=label,
                    y_score=out_proba,
                    multi_class="ovo",
                    labels=torch.arange(out_proba.shape[1]).tolist(),
                )
            else:
                assert out_proba.shape[1] == 2
                roc_auc = roc_auc_score(y_true=label, y_score=out_proba[:, 1])
        else:
            roc_auc = None

        # Per-sample metrics
        per_sample_acc = (out_pred == label).astype(np.int8).tolist()

        metrics = (metrics | {"balanced_accuracy": balanced_accuracy,
                              "roc_auc": roc_auc,
                              "per_sample_acc": per_sample_acc,
                              "per_sample_pred": out_pred.tolist(),
                              "per_sample_proba": out_proba.tolist() if out_proba is not None else None,
                              "label": label.tolist()})

    return metrics
