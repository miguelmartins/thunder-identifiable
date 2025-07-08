def download_pannuke(root_folder: str):
    import os

    from ..utils import download_from_url, unzip_file

    urls = [
        "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip",
        "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip",
        "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip",
    ]
    for url in urls:
        download_from_url(url, f"{root_folder}/content")
        unzip_file(f"{root_folder}/content", f"{root_folder}")

    # Renaming the folders to remove spaces
    os.rename(f"{root_folder}/Fold 1", f"{root_folder}/Fold1")
    os.rename(f"{root_folder}/Fold 2", f"{root_folder}/Fold2")
    os.rename(f"{root_folder}/Fold 3", f"{root_folder}/Fold3")


def create_splits_pannuke(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the PanNuke dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """

    import os

    from ..data_splits import check_dataset, init_dict, save_dict

    # Initializing dict
    pannuke_data_splits = init_dict()

    # Retrieving all images and labels
    pannuke_data_splits["train"]["images"] = "Fold1/images/fold1/images.npy"
    pannuke_data_splits["train"]["labels"] = "Fold1/masks/fold1/masks.npy"

    pannuke_data_splits["val"]["images"] = "Fold2/images/fold2/images.npy"
    pannuke_data_splits["val"]["labels"] = "Fold2/masks/fold2/masks.npy"

    pannuke_data_splits["test"]["images"] = "Fold3/images/fold3/images.npy"
    pannuke_data_splits["test"]["labels"] = "Fold3/masks/fold3/masks.npy"

    # Saving dict
    save_dict(
        pannuke_data_splits, os.path.join(base_folder, "data_splits", "pannuke.json")
    )
