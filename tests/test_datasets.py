import os
import tempfile

import pytest


def test_download_data(temp_env_dir):
    """Tests the download and split generation of the ocelot dataset."""
    # Download the dataset
    from thunder import download_datasets

    download_datasets("ocelot")

    # Check if the dataset is in the temp_env_dir
    dataset_path = os.path.join(temp_env_dir, "datasets/ocelot")
    assert os.path.exists(dataset_path), "Failed to download dataset."


def test_generate_splits(temp_env_dir):
    from thunder import generate_splits

    generate_splits("all")
    # Check if the splits are generated in the temp_env_dir
    splits_path = os.path.join(temp_env_dir, "datasets/data_splits/ocelot.json")
    assert os.path.exists(
        splits_path
    ), "Failed to generate splits (Splits file not found where expected)."
