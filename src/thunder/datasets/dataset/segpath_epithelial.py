import os


def download_segpath_epithelial(root_folder: str):
    from ..utils import download_from_url, untar_file

    url = "https://zenodo.org/api/records/7412731/files/panCK_fileinfo.csv/content"
    download_from_url(url, f"{root_folder}/content")
    os.rename(f"{root_folder}/content", f"{root_folder}/panCK_fileinfo.csv")

    url = "https://zenodo.org/api/records/7412731/files/panCK_Epithelium.tar.gz/content"
    download_from_url(url, f"{root_folder}/content")
    untar_file(f"{root_folder}/content", f"{root_folder}")


def create_splits_segpath_epithelial(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the SegPath-Epithelial dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """
    import os

    from ..data_splits import check_dataset, save_dict

    # Getting data splist dict
    segpath_epithelial_data_splits = segpath(
        base_folder, dataset_cfg, cell_type="epithelial"
    )

    # Checking dataset characteristics
    check_dataset(
        segpath_epithelial_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(
        segpath_epithelial_data_splits,
        os.path.join(base_folder, "data_splits", "segpath_epithelial.json"),
    )


def segpath(base_folder: str, dataset_cfg: dict, cell_type: str) -> None:
    """
    Generating data splits for the SegPath-derived datasets.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    :param cell_type: specific SegPath cell type.
    """
    import pandas as pd

    from ..data_splits import (check_dataset, compute_patches, init_dict,
                               save_dict, sorted_listdir)

    # Initializing dict
    segpath_data_splits = init_dict()

    # Retrieving all images and labels
    segpath_folder = os.path.join(base_folder, f"segpath_{cell_type}")
    segpath_files = sorted_listdir(segpath_folder)
    csv_segpath_files = [f for f in segpath_files if ".csv" in f]
    assert len(csv_segpath_files) == 1
    csv_filename = csv_segpath_files[0]

    df = pd.read_csv(os.path.join(segpath_folder, csv_filename))
    nb_filenames = 0
    for _, row in df.iterrows():
        nb_filenames += 1
        filename = row["filename"]
        data_split = row["train_val_test"]
        if "_HE.png" in filename:
            # Computing patches
            im_width, im_height = 984, 984
            images, labels = compute_patches(
                filename,
                filename.replace("_HE.png", "_mask.png"),
                im_height,
                im_width,
            )

            segpath_data_splits[data_split]["images"].extend(images)
            segpath_data_splits[data_split]["labels"].extend(labels)
    return segpath_data_splits
