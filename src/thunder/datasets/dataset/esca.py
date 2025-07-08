def download_esca(root_folder: str):
    from ..utils import download_from_url, untar_file

    urls = [
        "https://zenodo.org/api/records/7548828/files/VALSET1_UKK.tar/content",
        "https://zenodo.org/api/records/7548828/files/VALSET2_WNS.tar/content",
        "https://zenodo.org/api/records/7548828/files/VALSET3_TCGA.tar/content",
        "https://zenodo.org/api/records/7548828/files/VALSET4_CHA_FULL.tar/content",
    ]

    for url in urls:
        download_from_url(url, f"{root_folder}/content")
        untar_file(f"{root_folder}/content", f"{root_folder}")

    # Remove not necessary dotfile
    os.remove(f"{root_folder}/VALSET2_WNS/SH_MAG/._sh_mag.0.jpg")


def create_splits_esca(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the ESCA dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """

    import os

    from ...utils.constants import UtilsConstants
    from ...utils.utils import set_seed
    from ..data_splits import (check_dataset, create_few_shot_training_data,
                               get_data_from_set, init_dict, save_dict)

    # Setting the random seed
    set_seed(UtilsConstants.DEFAULT_SEED.value)

    # Initializing dict
    esca_data_splits = init_dict()

    # Retrieving all images and labels
    esca_folder = os.path.join(base_folder, "esca")
    # Following UNI (https://www.nature.com/articles/s41591-024-02857-3)
    # -> Training on UKK, WNS, TCGA and testing on CHA
    # + validating on UKK (UKK transferred from train to val)
    for folder in ["VALSET2_WNS", "VALSET3_TCGA"]:
        images, labels = get_data_from_set(esca_folder, folder, dataset_cfg)
        esca_data_splits["train"]["images"].extend(images)
        esca_data_splits["train"]["labels"].extend(labels)
    (
        esca_data_splits["val"]["images"],
        esca_data_splits["val"]["labels"],
    ) = get_data_from_set(esca_folder, "VALSET1_UKK", dataset_cfg)

    (
        esca_data_splits["test"]["images"],
        esca_data_splits["test"]["labels"],
    ) = get_data_from_set(esca_folder, "VALSET4_CHA_FULL", dataset_cfg)

    # Few-shot training data
    esca_data_splits = create_few_shot_training_data(esca_data_splits)

    # Checking dataset characteristics
    check_dataset(
        esca_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(esca_data_splits, os.path.join(base_folder, "data_splits", "esca.json"))
