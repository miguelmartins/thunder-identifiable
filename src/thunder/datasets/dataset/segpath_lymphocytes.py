import os


def download_segpath_lymphocytes(root_folder: str):
    from ..utils import download_from_url, untar_file

    url = "https://zenodo.org/api/records/7412529/files/CD3CD20_fileinfo.csv/content"
    download_from_url(url, f"{root_folder}/content")
    os.rename(f"{root_folder}/content", f"{root_folder}/CD3CD20_fileinfo.csv")

    url = (
        "https://zenodo.org/api/records/7412529/files/CD3CD20_Lymphocyte.tar.gz/content"
    )
    download_from_url(url, f"{root_folder}/content")
    untar_file(f"{root_folder}/content", f"{root_folder}")


def create_splits_segpath_lymphocytes(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the SegPath-Lymphocytes dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """
    import os

    from ..data_splits import check_dataset, save_dict
    from .segpath_epithelial import segpath

    # Getting data splist dict
    segpath_lymphocytes_data_splits = segpath(
        base_folder, dataset_cfg, cell_type="lymphocytes"
    )

    # Checking dataset characteristics
    check_dataset(
        segpath_lymphocytes_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(
        segpath_lymphocytes_data_splits,
        os.path.join(base_folder, "data_splits", "segpath_lymphocytes.json"),
    )
