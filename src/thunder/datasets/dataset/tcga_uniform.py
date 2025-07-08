def download_tcga_uniform(root_folder: str):
    from ..utils import download_from_url, unzip_file

    urls = [
        "https://zenodo.org/api/records/5889558/files/Adrenocortical_carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Bladder_Urothelial_Carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Brain_Lower_Grade_Glioma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Breast_invasive_carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Cholangiocarcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Colon_adenocarcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Esophageal_carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Glioblastoma_multiforme.zip/content",
        "https://zenodo.org/api/records/5889558/files/Head_and_Neck_squamous_cell_carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Kidney_Chromophobe.zip/content",
        "https://zenodo.org/api/records/5889558/files/Kidney_renal_clear_cell_carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Kidney_renal_papillary_cell_carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Liver_hepatocellular_carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Lung_adenocarcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Lung_squamous_cell_carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Mesothelioma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Ovarian_serous_cystadenocarcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Pancreatic_adenocarcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Pheochromocytoma_and_Paraganglioma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Prostate_adenocarcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Rectum_adenocarcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Sarcoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Skin_Cutaneous_Melanoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Stomach_adenocarcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Testicular_Germ_Cell_Tumors.zip/content",
        "https://zenodo.org/api/records/5889558/files/Thymoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Thyroid_carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Uterine_Carcinosarcoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Uterine_Corpus_Endometrial_Carcinoma.zip/content",
        "https://zenodo.org/api/records/5889558/files/Uveal_Melanoma.zip/content",
    ]

    for url in urls:
        download_from_url(url, f"{root_folder}/content")
        unzip_file(f"{root_folder}/content", f"{root_folder}")


def create_splits_tcga_uniform(base_folder: str, dataset_cfg: dict) -> None:
    """
    Generating data splits for the TCGA-UNIFORM dataset.

    :param base_folder: path to the main class_folder storing datasets.
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
    tcga_uniform_data_splits = init_dict()

    # Retrieving all images and labels
    tcga_uniform_folder = os.path.join(base_folder, "tcga_uniform")
    for class_folder in dataset_cfg.classes:
        ## folder "0" -> 0.5 μm/pixel
        curr_folder = os.path.join(tcga_uniform_folder, class_folder, "0/")
        slide_folders = sorted_listdir(curr_folder)
        slide_folders.sort()

        ## Avoiding train-test patient data leakage
        patient_id_to_folders = defaultdict(list)
        for slide_folder in slide_folders:
            slide_folder_split = slide_folder.split("-")
            patient_id = slide_folder_split[1] + slide_folder_split[2]
            patient_id_to_folders[patient_id].append(slide_folder)

        ## Random sampling of patients in train/val/test splits
        patient_ids = list(patient_id_to_folders.keys())
        random.shuffle(patient_ids)
        nb_patients = len(patient_ids)
        nb_train_val_patients = int(0.8 * nb_patients)
        ## val ~20% patients in train+val data
        nb_train_patients = int(0.8 * nb_train_val_patients)
        train_slide_folders = []
        val_slide_folders = []
        test_slide_folders = []
        for i in range(nb_patients):
            if i < nb_train_patients:
                train_slide_folders.extend(patient_id_to_folders[patient_ids[i]])
            elif i < nb_train_val_patients:
                val_slide_folders.extend(patient_id_to_folders[patient_ids[i]])
            else:
                test_slide_folders.extend(patient_id_to_folders[patient_ids[i]])

        train_class_images = get_class_images(
            train_slide_folders, curr_folder, class_folder
        )
        val_class_images = get_class_images(
            val_slide_folders, curr_folder, class_folder
        )
        test_class_images = get_class_images(
            test_slide_folders, curr_folder, class_folder
        )

        tcga_uniform_data_splits["train"]["images"].extend(train_class_images)
        tcga_uniform_data_splits["train"]["labels"].extend(
            [dataset_cfg.class_to_id[class_folder]] * len(train_class_images)
        )

        tcga_uniform_data_splits["val"]["images"].extend(val_class_images)
        tcga_uniform_data_splits["val"]["labels"].extend(
            [dataset_cfg.class_to_id[class_folder]] * len(val_class_images)
        )

        tcga_uniform_data_splits["test"]["images"].extend(test_class_images)
        tcga_uniform_data_splits["test"]["labels"].extend(
            [dataset_cfg.class_to_id[class_folder]] * len(test_class_images)
        )

    # Few-shot training data
    tcga_uniform_data_splits = create_few_shot_training_data(tcga_uniform_data_splits)

    # Checking dataset characteristics
    check_dataset(
        tcga_uniform_data_splits,
        dataset_cfg,
        base_folder,
    )

    # Saving dict
    save_dict(
        tcga_uniform_data_splits,
        os.path.join(base_folder, "data_splits", "tcga_uniform.json"),
    )


def get_class_images(image_folders: list, curr_folder: str, class_folder: str) -> list:
    """
    Retrieving all images for a given label class.

    :param image_folders: list of sub-folders containing images.
    :param curr_folder: path to the folder containing all the image sub-folders.
    :param class_folder: name of the class folder.
    :return: list of image filenames.
    """

    class_images = []
    for image_folder in image_folders:
        ims = sorted_listdir(os.path.join(curr_folder, image_folder))
        ims = [os.path.join(class_folder, "0/", image_folder, im) for im in ims]
        class_images.extend(ims)

    return class_images
