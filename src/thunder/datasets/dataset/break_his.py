def download_break_his(root_folder: str):
    from ..utils import download_from_url, untar_file

    download_from_url(
        "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz",
        f"{root_folder}/BreaKHis_v1.tar.gz",
    )
    untar_file(f"{root_folder}/BreaKHis_v1.tar.gz", f"{root_folder}")


def create_splits_break_his(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the BreakHis dataset.

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
    break_his_data_splits = init_dict()

    # Retrieving all images and labels
    break_his_folder = os.path.join(
        base_folder, "break_his", "BreaKHis_v1", "histology_slides", "breast"
    )
    ## Following EVA
    ## https://kaiko-ai.github.io/eva/main/datasets/breakhis/
    ## -> Keeping only classes with more than 7 patients (4 classes)
    ## https://github.com/kaiko-ai/eva/blob/2fea425c3deecc3281e1708dfddf5f234288a07c/src/eva/vision/data/datasets/classification/breakhis.py#L29-L45
    ## -> Same train/test split
    test_patient_ids = {
        "18842D",
        "19979",
        "15275",
        "15792",
        "16875",
        "3909",
        "5287",
        "16716",
        "2773",
        "5695",
        "16184CD",
        "23060CD",
        "21998CD",
        "21998EF",
    }
    ## + splitting train into train/val (val is ~20% train)
    ## -> sampling 10 patients out of 50 train patients
    ## with constraint to have reasonable distribution between 4 classes
    ## i.e. sampling repeated 10 times and chosen best set of 10 patients.
    val_patient_ids = {
        "23222AB",
        "13418DE",
        "11520",
        "19854C",
        "10926",
        "12312",
        "18842",
        "13412",
        "13200",
        "29960AB",
    }

    for group in ["benign", "malignant"]:
        group_folder = os.path.join(break_his_folder, group, "SOB")
        class_folders = sorted_listdir(group_folder)
        for class_folder in class_folders:
            if class_folder in dataset_cfg.classes:
                patient_folders = sorted_listdir(
                    os.path.join(group_folder, class_folder)
                )
                for patient_folder in patient_folders:
                    patient_id = patient_folder.split("-")[-1]
                    images = sorted_listdir(
                        os.path.join(group_folder, class_folder, patient_folder, "40X")
                    )
                    images = [
                        os.path.join(
                            "BreaKHis_v1",
                            "histology_slides",
                            "breast",
                            group,
                            "SOB",
                            class_folder,
                            patient_folder,
                            "40X",
                            im,
                        )
                        for im in images
                    ]
                    labels = [dataset_cfg.class_to_id[class_folder]] * len(images)

                    if patient_id in test_patient_ids:
                        break_his_data_splits["test"]["images"].extend(images)
                        break_his_data_splits["test"]["labels"].extend(labels)
                    elif patient_id in val_patient_ids:
                        break_his_data_splits["val"]["images"].extend(images)
                        break_his_data_splits["val"]["labels"].extend(labels)
                    else:
                        break_his_data_splits["train"]["images"].extend(images)
                        break_his_data_splits["train"]["labels"].extend(labels)

    # Few-shot training data
    break_his_data_splits = create_few_shot_training_data(break_his_data_splits)

    # Checking dataset characteristics
    check_dataset(
        break_his_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(
        break_his_data_splits,
        os.path.join(base_folder, "data_splits", "break_his.json"),
    )
