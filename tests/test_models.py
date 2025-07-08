import os
import tempfile

import pytest


def test_download_models(temp_env_dir):
    """Tests the download and split generation of the ocelot dataset."""
    # Download the dataset
    from thunder import download_models

    download_models("dinov2base")

    # Check if the dataset is in the temp_env_dir
    model_path = os.path.join(temp_env_dir, "pretrained_ckpts/dinov2base")
    assert os.path.exists(
        model_path
    ), "Failed to download model (Model directory not found where expected)."
