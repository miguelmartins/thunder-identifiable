def download_tcga_crc_msi(root_folder: str):
    from ..utils import download_from_url, unzip_file

    urls = [
        "https://zenodo.org/api/records/3832231/files/TRAIN.zip/content",
        "https://zenodo.org/api/records/3832231/files/TEST.zip/content",
    ]

    for url in urls:
        download_from_url(url, f"{root_folder}/content")
        unzip_file(f"{root_folder}/content", f"{root_folder}")


def create_splits_tcga_crc_msi(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the TCGA-CRC-MSI dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """
    import os
    import random
    from collections import defaultdict

    from ...utils.constants import UtilsConstants
    from ...utils.utils import set_seed
    from ..data_splits import (check_dataset, create_few_shot_training_data,
                               get_data_from_set, init_dict, save_dict)

    # Setting the random seed
    set_seed(UtilsConstants.DEFAULT_SEED.value)

    # Initializing dict
    tcga_crc_msi_data_splits = init_dict()

    # Retrieving all images and labels
    tcga_crc_msi_folder = os.path.join(base_folder, "tcga_crc_msi")
    (
        train_val_images,
        train_val_labels,
    ) = get_data_from_set(tcga_crc_msi_folder, "TRAIN", dataset_cfg)

    # Splitting train into train/val without patient leakage (val is ~20% patients)
    tcga_case_id_to_im = defaultdict(list)
    tcga_case_id_to_label = defaultdict(list)
    for i in range(len(train_val_images)):
        im, label = train_val_images[i], train_val_labels[i]
        case_id = im.split("/")[-1].split("-")
        case_id = case_id[1] + case_id[2]
        tcga_case_id_to_im[case_id].append(im)
        tcga_case_id_to_label[case_id].append(label)
    case_ids = tcga_case_id_to_im.keys()
    val_case_ids = random.sample(case_ids, int(0.2 * len(case_ids)))

    for case_id in case_ids:
        ims = tcga_case_id_to_im[case_id]
        labels = tcga_case_id_to_label[case_id]
        if case_id in val_case_ids:
            tcga_crc_msi_data_splits["val"]["images"].extend(ims)
            tcga_crc_msi_data_splits["val"]["labels"].extend(labels)
        else:
            tcga_crc_msi_data_splits["train"]["images"].extend(ims)
            tcga_crc_msi_data_splits["train"]["labels"].extend(labels)

    (
        tcga_crc_msi_data_splits["test"]["images"],
        tcga_crc_msi_data_splits["test"]["labels"],
    ) = get_data_from_set(tcga_crc_msi_folder, "TEST", dataset_cfg)

    # Few-shot training data
    tcga_crc_msi_data_splits = create_few_shot_training_data(tcga_crc_msi_data_splits)

    # Checking dataset characteristics
    check_dataset(
        tcga_crc_msi_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(
        tcga_crc_msi_data_splits,
        os.path.join(base_folder, "data_splits", "tcga_crc_msi.json"),
    )
