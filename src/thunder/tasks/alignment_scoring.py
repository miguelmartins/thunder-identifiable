import json
import os
import random
from collections import defaultdict

import numpy as np
import plotly
import plotly.express as px
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from ..utils.alignment_metrics import AlignmentMetrics
from ..utils.constants import ModelConstants
from ..utils.data import load_embeddings
from ..utils.utils import save_outputs


def alignment_scoring(
    base_embeddings_folder: str,
    model_A: str,
    dataset: str,
    max_nb_embeddings_comp: int,
    feats_A: torch.Tensor,
    res_folder: str,
    wandb_base_folder: str,
) -> None:
    """
    Computing aligment scores for the given model with all
    other models (whose embeddings have been pre-computed)
    for the given dataset.

    :param base_embeddings_folder: path to the base folder storing embeddings.
    :param model_A: name of the model to compare embeddings with others.
    :param dataset: name of the dataset.
    :param max_nb_embeddings_comp: maximum number of embeddings to compare.
    :param feats_A: array of embeddings for model_A and dataset.
    :param res_folder: folder to save results.
    :param wandb_base_folder: w&b folder.
    """

    # Pretrained models
    pretrained_models = ModelConstants.PRETRAINED_MODELS.value
    pretrained_models.remove(model_A)

    # Alignment metrics
    metrics = AlignmentMetrics.SUPPORTED_METRICS

    # Sampling and normalizing feats
    if len(feats_A) > max_nb_embeddings_comp:
        kept_indices = random.sample(range(len(feats_A)), max_nb_embeddings_comp)
        feats_A = feats_A[kept_indices]
    else:
        kept_indices = None
    feats_A = F.normalize(feats_A, dim=-1)

    # Computing alignment scores
    alignment_scores = defaultdict(dict)
    for model_B in tqdm(pretrained_models):
        feats_B_path = os.path.join(base_embeddings_folder, dataset, model_B)
        if os.path.exists(feats_B_path):
            # Loading feats for model B, sampling and normalizing
            feats_B, _ = load_embeddings(feats_B_path, ["train", "val"])
            feats_B = torch.Tensor(
                np.concatenate([feats_B["train"], feats_B["val"]], axis=0)
            )
            if kept_indices is not None:
                feats_B = feats_B[kept_indices]
            feats_B = F.normalize(feats_B, dim=-1)

            for metric in metrics:
                # Computing score
                kwargs = {}
                if "nn" in metric:
                    kwargs["topk"] = 10
                if "cca" in metric:
                    kwargs["cca_dim"] = 10
                if "kernel" in metric:
                    kwargs["dist"] = "sample"
                score = AlignmentMetrics.measure(metric, feats_A, feats_B, **kwargs)
                if type(score) == torch.Tensor:
                    score = score.item()
                alignment_scores[metric][model_B] = score

    # Plotting
    model_names = list(alignment_scores[metrics[0]].keys())
    alignment_score_histos = {}
    for metric in metrics:
        scores = [alignment_scores[metric][model_name] for model_name in model_names]
        fig = px.histogram(x=model_names, y=scores)
        fig.update_layout(xaxis_title="pretrained models")
        fig.update_layout(yaxis_title=f"{metric} alignment score of {model_A} with...")
        plot = wandb.Html(plotly.io.to_html(fig))
        alignment_score_histos[metric] = plot

    # Logging
    for metric in alignment_score_histos.keys():
        wandb.log({f"{wandb_base_folder}/{metric}": alignment_score_histos[metric]})
    save_outputs(res_folder, alignment_scores)
