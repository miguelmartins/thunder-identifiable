from test_multiple import EmbeddingsInstanceDataset, InstanceLearner
from typing import List, Tuple
import re
import os
import torch
import numpy as np


N = 10000
MAX_EPOCHS = 50
WEIGHT_DECAY = 0.05
LR = 1e-4
WARMUP = int(0.1 * MAX_EPOCHS)
N_RUNS = 100
NAMES = ["conch", "uni", "uni2h", "clipvitbasepatch32"]
EMBED = [512, 1024, 1536, 512]


def load_projector(name, version, embed):
    pass


def main():
    path_ = "/Users/miguelmartins/Projects/thunder-identifiable/logs"
    model_ = "clipvitbasepatch32_2025-09-30_17-52-18"
    version = 0
    sub_id = 0

    # log_path = f"{path_}/{model_}/version_{version}/{sub_id}"
    log_path = f"{path_}/{model_}/version_{version}"
    linear_projector = torch.nn.Linear(EMBED[0], N, bias=False)
    ckpt_path = f"{log_path}/checkpoints"
    checkpoint = os.listdir(ckpt_path)[0]

    # model = InstanceLearner.load_from_checkpoint(
    #     f"{ckpt_path}/{checkpoint}", hparams_file=f"{ckpt_path}/{sub_id}/hparams.yaml"
    # )
    ckpt = torch.load(f"{ckpt_path}/{checkpoint}")
    w = ckpt["state_dict"]["linear_projector.weight"]
    linear_projector.weight.data = w
    print(w.shape)

    w = w.cpu().numpy()
    s, v, d = np.linalg.svd(w)
    print(s.shape, v.shape, d.shape)


if __name__ == "__main__":
    main()
