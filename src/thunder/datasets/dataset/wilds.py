def download_wilds(root_folder: str):
    raise NotImplementedError("The WILDS dataset is not implemented yet.")


def create_splits_wilds(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the CAMELYON17-WILDS dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """

    import os

    from wilds import get_dataset

    from ...utils.constants import UtilsConstants
    from ...utils.utils import set_seed
    from ..data_splits import (check_dataset, create_few_shot_training_data,
                               init_dict, save_dict)

    # Setting the random seed
    set_seed(UtilsConstants.DEFAULT_SEED.value)

    # Initializing dict
    wilds_data_splits = init_dict()

    # Retrieving all images and labels
    wilds_folder = os.path.join(base_folder, "wilds")
    wilds_dataset = get_dataset(
        dataset="camelyon17", root_dir=os.path.join(wilds_folder, "data")
    )

    # Spliting images and labels
    for split in ["train", "id_val", "test", "val"]:
        subset = wilds_dataset.get_subset(split)
        filenames = [
            os.path.join("data", "camelyon17_v1.0", wilds_dataset._input_array[i])
            for i in subset.indices
        ]
        labels = [int(wilds_dataset._y_array[i].item()) for i in subset.indices]

        wilds_data_splits[split] = {
            "images": filenames,
            "labels": labels,
        }

    # Few-shot training data
    wilds_data_splits = create_few_shot_training_data(wilds_data_splits)

    # Checking dataset characteristics
    check_dataset(
        wilds_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(wilds_data_splits, os.path.join(base_folder, "data_splits", "wilds.json"))
