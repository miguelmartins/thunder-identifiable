import os
import tempfile

import pytest


def test_task_pre_computing_embeddings(temp_env_dir):
    """Tests the benchmark on break_his for the pre-computing embeddings task."""
    # Download the dataset
    from thunder import benchmark, download_datasets, download_models

    download_datasets(["break_his"], make_splits=True)
    download_models("phikon")

    benchmark("phikon", "break_his", "pre_computing_embeddings")

    # Check if results are generated in the temp_env_dir
    results_path = os.path.join(
        temp_env_dir, "embeddings/break_his/phikon/train/embeddings.h5"
    )
    assert os.path.exists(
        results_path
    ), "Failed to generate results (Embeddings file not found where expected)."


def test_custom_model(temp_env_dir):
    from thunder import benchmark
    from thunder.models import PretrainedModel

    class DINOv2Features(PretrainedModel):
        def __init__(self):
            super().__init__()

            import torch
            from torchvision import transforms

            self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.t = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                ]
            )
            self.name = "dinov2_vits14"
            self.emb_dim = 384

        def forward(self, x):
            feats = self.dinov2.forward_features(x)
            return feats["x_norm_clstoken"]

        def get_transform(self):
            return self.t

        def get_transform(self):
            return self.t

        def get_linear_probing_embeddings(self, x):
            x = self.dinov2.forward_features(x)
            return x["x_norm_clstoken"]

        def get_segmentation_embeddings(self, x):
            x = self.dinov2.forward_features(x)
            return x["x_norm_patchtokens"]

    benchmark(DINOv2Features(), "break_his", "image_retrieval")


def test_task_linear_probing(temp_env_dir):
    from thunder import benchmark

    benchmark("phikon", "break_his", "linear_probing", **{"adaptation.epochs": 1})


def test_task_knn(temp_env_dir):
    """Tests the benchmark on break_his for the knn task."""
    # Download the dataset
    from thunder import benchmark, download_datasets, download_models

    download_datasets(["break_his"], make_splits=True)
    download_models("phikon")

    benchmark("phikon", "break_his", "knn")

    # Check if results are generated in the temp_env_dir
    results_path = os.path.join(
        temp_env_dir, "outputs/res/break_his/phikon/knn/frozen/outputs.json"
    )
    assert os.path.exists(
        results_path
    ), "Failed to generate knn results (Output file not found where expected)."
