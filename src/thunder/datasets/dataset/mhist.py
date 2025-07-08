import logging


def download_mhist(root_folder: str):
    from ..utils import unzip_file

    logging.info(
        "In order to download the MHIST dataset, you will have to visit https://bmirds.github.io/MHIST/#accessing-dataset and fill in the data access form. After this, you should receive an email with links to download the required data. This procedure should only take you a couple of minutes."
    )

    # Ask user for input y/n
    user_input = (
        input("Do you wish to download the MHIST dataset? (y/n): ").strip().lower()
    )

    if user_input == "y":
        logging.info(
            f"We have created the MHIST folder. You should download the annotations.csv and images.zip following the links you received by email and place them in the folder: {root_folder}."
        )

        user_input = (
            input(
                "Have you placed annotations.csv and images.zip in the MHIST folder? (y/n): "
            )
            .strip()
            .lower()
        )

        if user_input == "y":
            # unzip images.zip
            unzip_file(
                f"{root_folder}/images.zip",
                f"{root_folder}",
            )
            os.remove(f"{root_folder}/images.zip")
        elif user_input == "n":
            pass
        else:
            logging.error("Invalid input. Please enter 'y' or 'n'.")
            return
    elif user_input == "n":
        # remove root folder
        os.rmdir(root_folder)
    else:
        logging.error("Invalid input. Please enter 'y' or 'n'.")
        return


def create_splits_mhist(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the MHIST dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """

    import os
    import random

    import pandas as pd

    from ...utils.constants import UtilsConstants
    from ...utils.utils import set_seed
    from ..data_splits import (check_dataset, create_few_shot_training_data,
                               init_dict, save_dict)

    # Setting the random seed
    set_seed(UtilsConstants.DEFAULT_SEED.value)

    # Initializing dict
    mhist_data_splits = init_dict()

    # Retrieving all images and labels
    mhist_folder = os.path.join(base_folder, "mhist")
    annotations_file = os.path.join(mhist_folder, "annotations.csv")
    df = pd.read_csv(annotations_file)

    for _, row in df.iterrows():
        image_name, label, partition = (
            row["Image Name"],
            row["Majority Vote Label"],
            row["Partition"],
        )
        assert partition in ["train", "test"]

        # Splitting train into train/val
        if partition == "train" and random.uniform(0, 1) < 0.2:
            partition = "val"

        mhist_data_splits[partition]["images"].append(
            os.path.join("images", image_name)
        )
        mhist_data_splits[partition]["labels"].append(dataset_cfg.class_to_id[label])

    # Few-shot training data
    mhist_data_splits = create_few_shot_training_data(mhist_data_splits)

    # Checking dataset characteristics
    check_dataset(
        mhist_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(mhist_data_splits, os.path.join(base_folder, "data_splits", "mhist.json"))
