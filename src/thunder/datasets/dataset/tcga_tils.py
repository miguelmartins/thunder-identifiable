def download_tcga_tils(root_folder: str):
    from ..utils import download_from_url, remove_symlinks, untar_file

    url = "https://zenodo.org/api/records/6604094/files/TCGA-TILs.tar.gz/content"
    download_from_url(url, f"{root_folder}/content")
    untar_file(f"{root_folder}/content", f"{root_folder}")
    remove_symlinks(root_folder)


def create_splits_tcga_tils(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the TCGA-TILS dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """

    import os

    from ...utils.constants import UtilsConstants
    from ...utils.utils import set_seed
    from ..data_splits import (check_dataset, create_few_shot_training_data,
                               init_dict, save_dict, sorted_listdir)

    # Setting the random seed
    set_seed(UtilsConstants.DEFAULT_SEED.value)

    # Initializing dict
    tcga_tils_data_splits = init_dict()

    # Retrieving all images and labels
    tcga_tils_folder = os.path.join(
        base_folder, "tcga_tils", "TCGA-TILs", "images-tcga-tils"
    )
    folders = sorted_listdir(tcga_tils_folder)
    for folder in folders:
        data_splits = sorted_listdir(os.path.join(tcga_tils_folder, folder))
        data_splits.sort()
        assert data_splits == ["test", "train", "val"]
        for data_split in data_splits:
            for image_class in dataset_cfg.classes:
                images_folder = os.path.join(
                    tcga_tils_folder, folder, data_split, image_class
                )
                if os.path.exists(images_folder):
                    images = sorted_listdir(images_folder)
                    images = [
                        os.path.join(
                            "TCGA-TILs",
                            "images-tcga-tils",
                            folder,
                            data_split,
                            image_class,
                            im,
                        )
                        for im in images
                    ]
                    labels = [dataset_cfg.class_to_id[image_class]] * len(images)
                    tcga_tils_data_splits[data_split]["images"].extend(images)
                    tcga_tils_data_splits[data_split]["labels"].extend(labels)

    # Few-shot training data
    tcga_tils_data_splits = create_few_shot_training_data(tcga_tils_data_splits)

    # Checking dataset characteristics
    check_dataset(
        tcga_tils_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(
        tcga_tils_data_splits,
        os.path.join(base_folder, "data_splits", "tcga_tils.json"),
    )
