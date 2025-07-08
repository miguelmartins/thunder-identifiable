def download_bach(root_folder: str):
    from ..utils import (download_from_url, remove_symlinks, ungzip_file,
                         untar_file, unzip_file)

    urls = [
        "https://zenodo.org/api/records/3632035/files/ICIAR2018_BACH_Challenge.zip/content",
        "https://zenodo.org/api/records/3632035/files/ICIAR2018_BACH_Challenge_TestDataset.zip/content",
    ]

    for url in urls:
        download_from_url(url, f"{root_folder}/content")
        unzip_file(f"{root_folder}/content", f"{root_folder}")


def create_splits_bach(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the BACH dataset.

    :param base_folder: path to the main folder storing datasets.
    :param dataset_cfg: dataset-specific config.
    """
    import math
    import os
    from collections import defaultdict
    from pathlib import Path

    import pandas as pd

    from ...utils.constants import UtilsConstants
    from ...utils.utils import set_seed
    from ..data_splits import (check_dataset, create_few_shot_training_data,
                               init_dict, save_dict, sorted_listdir)

    # Patient details are derived from ICIAR2018_BACH_dataset_micro_patient.xlsx
    # downloaded from https://www.dropbox.com/scl/fo/efjnzcjydcjc4ebz1py7n/AIQyydtZTQcAFlLRxsEQhCU?dl=0&e=5&preview=ICIAR2018_BACH_dataset_micro_patient.xlsx&rlkey=w5ebtg9dlwrk57663chvodzy6&st=k0dfgjck
    # and adapted.
    patients = {
        "Histology": [
            "b032.tif",
            "b033.tif",
            "b034.tif",
            "b035.tif",
            "b036.tif",
            "b037.tif",
            "b038.tif",
            "b039.tif",
            "b040.tif",
            "b041.tif",
            "b042.tif",
            "b043.tif",
            "b044.tif",
            "b045.tif",
            "b046.tif",
            "b047.tif",
            "b048.tif",
            "b049.tif",
            "b050.tif",
            "b051.tif",
            "b052.tif",
            "b053.tif",
            "b054.tif",
            "b055.tif",
            "b056.tif",
            "b057.tif",
            "b058.tif",
            "b059.tif",
            "b060.tif",
            "b061.tif",
            "b062.tif",
            "b063.tif",
            "b064.tif",
            "b065.tif",
            "b066.tif",
            "b067.tif",
            "b068.tif",
            "b069.tif",
            "b070.tif",
            "b071.tif",
            "b072.tif",
            "b073.tif",
            "b074.tif",
            "b075.tif",
            "b076.tif",
            "b077.tif",
            "b078.tif",
            "b079.tif",
            "b080.tif",
            "b081.tif",
            "b082.tif",
            "b083.tif",
            "b084.tif",
            "b085.tif",
            "b086.tif",
            "b087.tif",
            "b088.tif",
            "b089.tif",
            "b090.tif",
            "is040.tif",
            "is041.tif",
            "is042.tif",
            "is043.tif",
            "is044.tif",
            "is045.tif",
            "is046.tif",
            "is047.tif",
            "is048.tif",
            "is049.tif",
            "is050.tif",
            "is051.tif",
            "is052.tif",
            "is053.tif",
            "is054.tif",
            "is055.tif",
            "is056.tif",
            "is057.tif",
            "is058.tif",
            "is059.tif",
            "is060.tif",
            "is061.tif",
            "is062.tif",
            "is063.tif",
            "is064.tif",
            "is065.tif",
            "is066.tif",
            "is067.tif",
            "is068.tif",
            "is069.tif",
            "iv041.tif",
            "iv042.tif",
            "iv043.tif",
            "iv044.tif",
            "iv045.tif",
            "iv046.tif",
            "iv047.tif",
            "iv048.tif",
            "iv049.tif",
            "iv050.tif",
            "iv051.tif",
            "iv052.tif",
            "iv053.tif",
            "iv054.tif",
            "iv055.tif",
            "iv056.tif",
            "iv057.tif",
            "iv058.tif",
            "iv059.tif",
            "iv060.tif",
            "iv061.tif",
            "iv062.tif",
            "iv063.tif",
            "iv064.tif",
            "iv065.tif",
            "iv066.tif",
            "iv067.tif",
            "iv068.tif",
            "iv069.tif",
            "iv070.tif",
            "iv071.tif",
            "iv072.tif",
            "iv073.tif",
            "n046.tif",
            "n047.tif",
            "n048.tif",
            "n049.tif",
            "n050.tif",
            "n051.tif",
            "n052.tif",
            "n053.tif",
            "n054.tif",
            "n055.tif",
            "n056.tif",
            "n057.tif",
            "n058.tif",
            "n059.tif",
            "n060.tif",
            "n061.tif",
            "n062.tif",
            "n063.tif",
            "n064.tif",
            "n065.tif",
            "n066.tif",
            "n067.tif",
            "n068.tif",
        ],
        "Label": [
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "Benign",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "InSitu",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Invasive",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
            "Normal",
        ],
        "Patient": [
            11,
            11,
            11,
            11,
            12,
            12,
            12,
            13,
            13,
            15,
            16,
            16,
            16,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            18,
            19,
            20,
            20,
            20,
            21,
            21,
            21,
            21,
            21,
            21,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            23,
            25,
            26,
            22,
            24,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            27,
            27,
            28,
            28,
            29,
            25,
            26,
            30,
            30,
            31,
            31,
            31,
            31,
            32,
            32,
            33,
            33,
            22,
            19,
            19,
            34,
            34,
            35,
            36,
            37,
            37,
            38,
            38,
            39,
            39,
            14,
            14,
            14,
            16,
            16,
            16,
            16,
            16,
            16,
            17,
            17,
            17,
            17,
            18,
            18,
            18,
            18,
            18,
            22,
            22,
            20,
            21,
            21,
            21,
            21,
            21,
        ],
    }

    # Setting the random seed
    set_seed(UtilsConstants.DEFAULT_SEED.value)

    # Initializing dict
    bach_data_splits = init_dict()

    # Retrieving all images and labels
    bach_folder = os.path.join(
        base_folder, "bach", "ICIAR2018_BACH_Challenge", "Photos"
    )
    bach_all_images = []
    bach_all_labels = []

    for folder in dataset_cfg.classes:
        images = sorted_listdir(os.path.join(bach_folder, folder))
        images = [s for s in images if ".tif" in s]
        images.sort()
        images = [
            os.path.join("ICIAR2018_BACH_Challenge", "Photos", folder, s)
            for s in images
        ]
        assert len(images) == 100
        bach_all_images.extend(images)
        bach_all_labels.extend([dataset_cfg.class_to_id[folder]] * len(images))

    # Following eva splitting + splitting eva train into train and val (val is ~20% of train set)
    # (https://github.com/kaiko-ai/eva/blob/71a2e1891c45156e5a250699fbbec5e741604555/src/eva/vision/data/datasets/classification/bach.py#L19-L38)
    BACH_TRAIN_INDEX_RANGES = [
        (0, 15),
        (25, 41),
        (59, 60),
        (90, 110),
        (120, 139),
        (169, 200),
        (210, 240),
        (258, 260),
        (273, 305),
        (315, 345),
        (368, 380),
        (390, 400),
    ]
    BACH_VAL_INDEX_RANGES = [
        (15, 25),
        (110, 120),
        (200, 210),
        (305, 315),
        (380, 390),
    ]
    BACH_TEST_INDEX_RANGES = [
        (41, 59),
        (60, 90),
        (139, 169),
        (240, 258),
        (260, 273),
        (345, 368),
    ]
    bach_train_indices = [
        index for start, end in BACH_TRAIN_INDEX_RANGES for index in range(start, end)
    ]
    bach_val_indices = [
        index for start, end in BACH_VAL_INDEX_RANGES for index in range(start, end)
    ]
    bach_test_indices = [
        index for start, end in BACH_TEST_INDEX_RANGES for index in range(start, end)
    ]

    # Spliting images and labels
    bach_data_splits["train"]["images"] = [
        bach_all_images[i] for i in bach_train_indices
    ]
    bach_data_splits["train"]["labels"] = [
        bach_all_labels[i] for i in bach_train_indices
    ]

    bach_data_splits["val"]["images"] = [bach_all_images[i] for i in bach_val_indices]
    bach_data_splits["val"]["labels"] = [bach_all_labels[i] for i in bach_val_indices]

    bach_data_splits["test"]["images"] = [bach_all_images[i] for i in bach_test_indices]
    bach_data_splits["test"]["labels"] = [bach_all_labels[i] for i in bach_test_indices]

    # Checking there is no patient leakage
    df = pd.DataFrame(patients)

    def get_bach_patients_set(df, bach_images):
        bach_patients = set()
        for image in bach_images:
            histology = image.split("/")[-1]
            if histology in df["Histology"].values:
                patient_id = df[df["Histology"] == histology]["Patient"].item()
                bach_patients.add(int(patient_id))
        return bach_patients

    bach_train_patients = get_bach_patients_set(df, bach_data_splits["train"]["images"])
    bach_val_patients = get_bach_patients_set(df, bach_data_splits["val"]["images"])
    bach_test_patients = get_bach_patients_set(df, bach_data_splits["test"]["images"])

    train_inter_test = bach_train_patients.intersection(bach_test_patients)
    assert (
        len(train_inter_test) == 0
    ), "There is patient leakage between train and test sets."
    train_inter_val = bach_train_patients.intersection(bach_val_patients)
    assert (
        len(train_inter_val) == 0
    ), "There is patient leakage between train and val sets."
    val_inter_test = bach_val_patients.intersection(bach_test_patients)
    assert (
        len(val_inter_test) == 0
    ), "There is patient leakage between val and test sets."

    # Few-shot training data
    bach_data_splits = create_few_shot_training_data(bach_data_splits)

    # Checking dataset characteristics
    check_dataset(
        bach_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(bach_data_splits, os.path.join(base_folder, "data_splits", "bach.json"))
