def download_ccrcc(root_folder: str):
    from ..utils import download_from_url, unzip_file

    download_from_url(
        "https://zenodo.org/api/records/7898308/files/tissue_classification.zip/content",
        f"{root_folder}/content",
    )
    unzip_file(f"{root_folder}/content", f"{root_folder}")


def create_splits_ccrcc(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the CCRCC dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """
    import os
    import random
    from collections import defaultdict

    from ...utils.constants import UtilsConstants
    from ...utils.utils import set_seed
    from ..data_splits import (check_dataset, create_few_shot_training_data,
                               init_dict, save_dict, sorted_listdir)

    # Setting the random seed
    set_seed(UtilsConstants.DEFAULT_SEED.value)

    # Initializing dict
    ccrcc_data_splits = init_dict()

    # Retrieving all images and labels
    ccrcc_folder = os.path.join(base_folder, "ccrcc", "tissue_classification")
    for subfolder in dataset_cfg.classes:
        images = sorted_listdir(os.path.join(ccrcc_folder, subfolder))

        # Following UNI (https://www.nature.com/articles/s41591-024-02857-3)
        # -> Training on TCGA and testing on Helsinki images
        # + splitting train into train/val without patient leakage (val is ~20% patients)
        tcga_case_id_to_im = defaultdict(list)
        for im in images:
            if "TCGA" in im:
                case_id = im.split("_")[0].split("-")
                case_id = case_id[1] + case_id[2]
                tcga_case_id_to_im[case_id].append(im)
        case_ids = tcga_case_id_to_im.keys()
        val_case_ids = random.sample(case_ids, int(0.2 * len(case_ids)))

        train_images, val_images = [], []
        for case_id in case_ids:
            ims = [
                os.path.join("tissue_classification", subfolder, s)
                for s in tcga_case_id_to_im[case_id]
            ]
            if case_id in val_case_ids:
                val_images.extend(ims)
            else:
                train_images.extend(ims)

        # Test data
        test_images = [
            os.path.join("tissue_classification", subfolder, s)
            for s in images
            if "TCGA" not in s
        ]

        ccrcc_data_splits["train"]["images"].extend(train_images)
        ccrcc_data_splits["train"]["labels"].extend(
            [dataset_cfg.class_to_id[subfolder]] * len(train_images)
        )
        ccrcc_data_splits["val"]["images"].extend(val_images)
        ccrcc_data_splits["val"]["labels"].extend(
            [dataset_cfg.class_to_id[subfolder]] * len(val_images)
        )
        ccrcc_data_splits["test"]["images"].extend(test_images)
        ccrcc_data_splits["test"]["labels"].extend(
            [dataset_cfg.class_to_id[subfolder]] * len(test_images)
        )

    # Few-shot training data
    ccrcc_data_splits = create_few_shot_training_data(ccrcc_data_splits)

    # Checking dataset characteristics
    check_dataset(
        ccrcc_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(ccrcc_data_splits, os.path.join(base_folder, "data_splits", "ccrcc.json"))
