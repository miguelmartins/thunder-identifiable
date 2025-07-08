def download_patch_camelyon(root_folder: str):
    from ..utils import download_from_url, ungzip_file

    urls = [
        (
            "https://zenodo.org/api/records/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz/content",
            "camelyonpatch_level_2_split_train_x.h5",
        ),
        (
            "https://zenodo.org/api/records/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz/content",
            "camelyonpatch_level_2_split_train_y.h5",
        ),
        (
            "https://zenodo.org/api/records/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz/content",
            "camelyonpatch_level_2_split_valid_x.h5",
        ),
        (
            "https://zenodo.org/api/records/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz/content",
            "camelyonpatch_level_2_split_valid_y.h5",
        ),
        (
            "https://zenodo.org/api/records/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz/content",
            "camelyonpatch_level_2_split_test_x.h5",
        ),
        (
            "https://zenodo.org/api/records/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz/content",
            "camelyonpatch_level_2_split_test_y.h5",
        ),
    ]

    for url, name in urls:
        download_from_url(url, f"{root_folder}/content")
        ungzip_file(f"{root_folder}/content", f"{root_folder}/{name}")


def create_splits_patch_camelyon(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the PatchCamelyon dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """
    import os
    import random
    from collections import defaultdict

    import h5py
    import numpy as np
    import pandas as pd

    from ...utils.constants import UtilsConstants
    from ...utils.utils import set_seed
    from ..data_splits import check_dataset, init_dict, save_dict

    # Setting the random seed
    set_seed(UtilsConstants.DEFAULT_SEED.value)

    # Initializing dict
    patch_camelyon_data_splits = init_dict()

    # Retrieving all images and labels
    for split in ["train", "val", "test"]:
        fill_patch_camelyon_data_splits(patch_camelyon_data_splits, split)

    # Few-shot training data
    nb_sets_per_nb_shot = 1000
    ## Loading training labels
    train_labels_path = os.path.join(
        os.environ["THUNDER_BASE_DATA_FOLDER"],
        "datasets",
        "patch_camelyon",
        patch_camelyon_data_splits["train"]["labels"],
    )
    train_labels = h5py.File(train_labels_path, "r")
    train_labels = np.array(train_labels.get("y")).reshape((-1))

    ## Creating few-shot sets
    label2images = defaultdict(list)
    for i in range(len(train_labels)):
        label2images[train_labels[i].item()].append(i)

    for nb_shots in patch_camelyon_data_splits["train_few_shot"].keys():
        for _ in range(nb_sets_per_nb_shot):
            few_shot_images = []
            few_shot_labels = []
            for label in label2images.keys():
                rd_images = random.sample(label2images[label], int(nb_shots))
                few_shot_images.extend(rd_images)
                few_shot_labels.extend([label for _ in range(int(nb_shots))])

            patch_camelyon_data_splits["train_few_shot"][nb_shots]["images"].append(
                few_shot_images
            )
            patch_camelyon_data_splits["train_few_shot"][nb_shots]["labels"].append(
                few_shot_labels
            )

    # Saving dict
    save_dict(
        patch_camelyon_data_splits,
        os.path.join(base_folder, "data_splits", "patch_camelyon.json"),
    )


def fill_patch_camelyon_data_splits(data_split_dict: dict, split: str) -> None:
    """
    Filling the data split dictionary for a given split.

    :param data_split_dict: data split dictionary.
    :param split: name of the split.
    """
    data_split_dict[split][
        "images"
    ] = f"camelyonpatch_level_2_split_{split if split != 'val' else 'valid'}_x.h5"
    data_split_dict[split][
        "labels"
    ] = f"camelyonpatch_level_2_split_{split if split != 'val' else 'valid'}_y.h5"
