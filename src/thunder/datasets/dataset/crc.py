def download_crc(root_folder: str):
    from ..utils import download_from_url, unzip_file

    download_from_url(
        "https://zenodo.org/api/records/1214456/files/NCT-CRC-HE-100K.zip/content",
        f"{root_folder}/content",
    )

    unzip_file(f"{root_folder}/content", f"{root_folder}")

    download_from_url(
        "https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content",
        f"{root_folder}/content",
    )
    unzip_file(f"{root_folder}/content", f"{root_folder}")


def create_splits_crc(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the CRC-100K dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """

    import os
    import random

    import numpy as np

    from ...utils.constants import UtilsConstants
    from ...utils.utils import set_seed
    from ..data_splits import (check_dataset, create_few_shot_training_data,
                               get_data_from_set, init_dict, save_dict)

    # Setting the random seed
    set_seed(UtilsConstants.DEFAULT_SEED.value)

    # Initializing dict
    crc_data_splits = init_dict()

    # Retrieving all images and labels
    crc_folder = os.path.join(base_folder, "crc")
    (
        train_val_images,
        train_val_labels,
    ) = get_data_from_set(crc_folder, "NCT-CRC-HE-100K", dataset_cfg)
    # Sampling val samples
    train_val_images, train_val_labels = (
        np.array(train_val_images),
        np.array(train_val_labels),
    )
    val_idx = random.sample(
        range(len(train_val_images)), int(0.2 * len(train_val_images))
    )
    val_mask = np.full(len(train_val_images), False, dtype=bool)
    val_mask[val_idx] = True
    crc_data_splits["val"]["images"], crc_data_splits["val"]["labels"] = (
        train_val_images[val_mask].tolist(),
        train_val_labels[val_mask].tolist(),
    )
    crc_data_splits["train"]["images"], crc_data_splits["train"]["labels"] = (
        train_val_images[~val_mask].tolist(),
        train_val_labels[~val_mask].tolist(),
    )
    # Test samples
    (
        crc_data_splits["test"]["images"],
        crc_data_splits["test"]["labels"],
    ) = get_data_from_set(crc_folder, "CRC-VAL-HE-7K", dataset_cfg)

    # Checking dataset characteristics
    check_dataset(
        crc_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Few-shot training data
    crc_data_splits = create_few_shot_training_data(crc_data_splits)

    # Saving dict
    save_dict(crc_data_splits, os.path.join(base_folder, "data_splits", "crc.json"))
