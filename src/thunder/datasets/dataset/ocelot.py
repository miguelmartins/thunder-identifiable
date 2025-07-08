def download_ocelot(root_folder: str):
    from ..utils import download_from_url, unzip_file

    urt = "https://zenodo.org/api/records/8417503/files/ocelot2023_v1.0.1.zip/content"
    download_from_url(urt, f"{root_folder}/content")
    unzip_file(f"{root_folder}/content", f"{root_folder}")


def create_splits_ocelot(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the OCELOT dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """

    import os

    from ..data_splits import (check_dataset, compute_patches, init_dict,
                               save_dict, sorted_listdir)

    # Initializing dict
    ocelot_data_splits = init_dict()

    # Retrieving all images and labels
    ocelot_folder = os.path.join(base_folder, "ocelot", "ocelot2023_v1.0.1")
    for data_split in ["train", "val", "test"]:
        filenames = sorted_listdir(
            os.path.join(ocelot_folder, "images", data_split, "tissue")
        )
        for filename in filenames:
            # Computing patches
            im_width, im_height = 1024, 1024
            images, labels = compute_patches(
                os.path.join(
                    "ocelot2023_v1.0.1",
                    "images",
                    data_split,
                    "tissue",
                    filename,
                ),
                os.path.join(
                    "ocelot2023_v1.0.1",
                    "annotations",
                    data_split,
                    "tissue",
                    filename.replace(".jpg", ".png"),
                ),
                im_height,
                im_width,
            )

            ocelot_data_splits[data_split]["images"].extend(images)
            ocelot_data_splits[data_split]["labels"].extend(labels)

    # Checking dataset characteristics
    check_dataset(
        ocelot_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(
        ocelot_data_splits, os.path.join(base_folder, "data_splits", "ocelot.json")
    )
