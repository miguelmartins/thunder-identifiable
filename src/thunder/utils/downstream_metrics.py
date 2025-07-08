from typing import Callable, Union

import numpy as np
import torch
from scipy.stats import bootstrap
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             jaccard_score, roc_auc_score)

from ..utils.constants import UtilsConstants


def compute_metric(
    label: Union[torch.Tensor, np.array],
    out: Union[torch.Tensor, np.array],
    metric_fn: Callable,
    label_indices: list = None,
    compute_ci: bool = True,
) -> dict:
    """
    Computing the metric score and associated bootstrap 95% confidence interval.
    :param label: tensor of ground-truth labels.
    :param out: tensor of outputs.
    :param metric_fn: callable computing metric score from outputs and ground truth.
    :param label_indices: list of label indices if required to compute boostrap confidence interval.
    :param compute_ci: whether to compute confidence interval.
    :return dictionary containing metric score, lower and upper bounds of confidence interval.
    """
    # Metric score
    metric_score = metric_fn(label, out)

    # 95% bootstrap confidence interval
    if compute_ci:
        if label_indices is None:
            metric_fn_inputs = (label, out)
            bootstrap_metric_fn = lambda y, y_pred: metric_fn(y, y_pred)
        else:
            metric_fn_inputs = (label_indices,)
            bootstrap_metric_fn = lambda label_indices: metric_fn(
                label[label_indices], out[label_indices]
            )

        bootstrap_res = bootstrap(
            metric_fn_inputs,
            bootstrap_metric_fn,
            n_resamples=3000,
            confidence_level=0.95,
            method="percentile",
            paired=True,
            rng=np.random.default_rng(UtilsConstants.DEFAULT_SEED.value),
        )
        ci = bootstrap_res.confidence_interval
        ci_low = ci.low
        ci_high = ci.high
    else:
        ci_low, ci_high = None, None

    return {
        "metric_score": metric_score,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def compute_metrics(
    out_proba: Union[torch.Tensor, np.array],
    out_pred: Union[torch.Tensor, np.array],
    label: Union[torch.Tensor, np.array],
    is_segmentation: bool = False,
    compute_ci: bool = True,
) -> dict:
    """
    Computing performance metrics.
    :param out_proba: tensor of predicted probabilities.
    :param out_pred: tensor of predicted labels.
    :param label: tensor of ground-truth labels.
    :param is_segmentation: whether we are doing semgnetation.
    :param compute_ci: whether to compute confidence intervals.
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

    # Accuracy, f1-score, jaccard index
    accuracy = compute_metric(
        label,
        out_pred,
        lambda y, y_pred: accuracy_score(y_true=y, y_pred=y_pred),
        compute_ci=compute_ci,
    )
    f1 = compute_metric(
        label,
        out_pred,
        lambda y, y_pred: f1_score(y_true=y, y_pred=y_pred, average="macro"),
        compute_ci=compute_ci,
    )
    jaccard = compute_metric(
        label,
        out_pred,
        lambda y, y_pred: jaccard_score(y_true=y, y_pred=y_pred, average="macro"),
        compute_ci=compute_ci,
    )

    metrics = {"f1": f1, "accuracy": accuracy, "jaccard": jaccard}

    if not is_segmentation:
        # Balanced accuracy
        balanced_accuracy = compute_metric(
            label,
            out_pred,
            lambda y, y_pred: balanced_accuracy_score(y_true=y, y_pred=y_pred),
            compute_ci=compute_ci,
        )

        # ROC AUC
        if out_proba is not None:
            if out_proba.shape[1] > 2:
                metric_fn = lambda y, y_proba: roc_auc_score(
                    y_true=y,
                    y_score=y_proba,
                    multi_class="ovo",
                    labels=torch.arange(out_proba.shape[1]).tolist(),
                )
                roc_auc = compute_metric(
                    label,
                    out_proba,
                    metric_fn,
                    label_indices=np.arange(len(label)),
                    compute_ci=compute_ci,
                )
            else:
                assert out_proba.shape[1] == 2
                roc_auc = compute_metric(
                    label,
                    out_proba,
                    lambda y, y_proba: roc_auc_score(y_true=y, y_score=y_proba[:, 1]),
                    label_indices=np.arange(len(label)),
                    compute_ci=compute_ci,
                )
        else:
            roc_auc = None

        # Per-sample accuracy
        per_sample_acc = (out_pred == label).astype(np.int8).tolist()

        # GT labels, per-sample predictions and probas added to computed metrics
        metrics = metrics | {
            "balanced_accuracy": balanced_accuracy,
            "roc_auc": roc_auc,
            "per_sample_accuracy": per_sample_acc,
            "per_sample_pred": out_pred.tolist(),
            "per_sample_proba": out_proba.tolist() if out_proba is not None else None,
            "label": label.tolist(),
        }

    return metrics
