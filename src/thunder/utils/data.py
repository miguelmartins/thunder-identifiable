import json
import os

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

from ..utils.constants import DatasetConstants


def get_data(dataset_name: str, base_data_folder: str) -> dict:
    """
    Getting data (train and test) for the specific dataset.
    :param dataset_name: name of the dataset to retrieve information from.
    :param base_data_folder: base path where data is stored.
    :return data dictionary.
    """
    assert (
        dataset_name in DatasetConstants.DATASETS.value
    ), f"{dataset_name} is not within the list of available datasets: {DatasetConstants.DATASETS.value}."

    # Reading json data file
    data_path = os.path.join(base_data_folder, "data_splits", f"{dataset_name}.json")
    with open(data_path, "r") as file:
        data = json.load(file)

    return data


def h5_to_np(h5_array: h5py._hl.files.File) -> np.array:
    """
    Converting an array stored with h5 format into a numpy array.
    :param h5_array: array stored with h5 format.
    :return numpy array.
    """
    return np.concatenate(
        [np.array(h5_array[str(i)])[None] for i in range(len(h5_array.keys()))]
    )


def load_embeddings(embeddings_folder: str, splits: list):
    """
    Loading embeddings and associated labels.
    :param embeddings_folder: folder storing embeddings.
    :param splits: list of data splits to load.
    :return dict of embeddings and labels.
    """
    embs = {}
    labels = {}
    for split in splits:
        h5_embs = h5py.File(os.path.join(embeddings_folder, split, "embeddings.h5"))
        h5_labels = h5py.File(os.path.join(embeddings_folder, split, "labels.h5"))
        embs[split] = h5_to_np(h5_embs)
        labels[split] = h5_to_np(h5_labels)
    return embs, labels


class PatchDataset(Dataset):
    def __init__(
        self,
        images: list,
        labels: list,
        transform: transforms.Compose,
        task_type: str,
        dataset_name: str,
        base_data_folder: str,
        embeddings_folder: str,
        image_pre_loading: bool = False,
        embedding_pre_loading: bool = False,
    ) -> None:
        """
        Initializing dataset information.
        :param images: list of images.
        :param labels: list of ground-truth labels.
        :param transform: transformation to apply to images.
        :param task_type: type of task (classification, segmentation).
        :param dataset_name: name of the loaded dataset.
        :param base_data_folder: path to the base folder storing data.
        :param embeddings_folder: folder storing embeddings.
        :param image_pre_loading: whether to pre-load all images.
        :param embedding_pre_loading: whether to pre-load all embeddings.
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.base_data_folder = base_data_folder
        self.image_pre_loading = image_pre_loading
        self.embedding_pre_loading = embedding_pre_loading

        if dataset_name == "patch_camelyon" and not embedding_pre_loading:
            self.images = h5py.File(
                os.path.join(self.base_data_folder, self.dataset_name, self.images), "r"
            )
            self.labels = h5py.File(
                os.path.join(self.base_data_folder, self.dataset_name, self.labels), "r"
            )
            self.images = np.array(self.images.get("x"))
            self.labels = np.array(self.labels.get("y")).reshape((-1))
        elif dataset_name == "pannuke":
            self.images = np.load(
                os.path.join(
                    self.base_data_folder,
                    self.dataset_name,
                    self.images,
                ),
                "r",
            )
            masks = np.load(
                os.path.join(
                    self.base_data_folder,
                    self.dataset_name,
                    self.labels,
                ),
                "r",
            )

            # Adapted from https://github.com/TIO-IKIM/CellViT/blob/main/cell_segmentation/datasets/prepare_pannuke.py#L61-L64
            self.labels = np.zeros((masks.shape[0], 256, 256)).astype(np.int32)
            for i in range(masks.shape[0]):
                for j in range(5):
                    layer_res = ((j + 1) * np.clip(masks[i, :, :, j], 0, 1)).astype(
                        np.int32
                    )
                    self.labels[i] = np.where(layer_res != 0, layer_res, self.labels[i])
            self.labels = torch.Tensor(self.labels).to(torch.int64)

        if image_pre_loading:
            for i in range(len(self.images)):
                image, viz_image = self.image_loading(i)
                self.images[i] = (image, viz_image)

        if embedding_pre_loading:
            self.embeddings_path = os.path.join(embeddings_folder, "embeddings.h5")
            self.labels_path = os.path.join(embeddings_folder, "labels.h5")

            with h5py.File(self.labels_path, "r") as lab_h5:
                self._len = len(lab_h5)

            self.labels = None
        else:
            self._len = len(self.labels)

    def __len__(self) -> int:
        """
        Returning the length of the dataset.
        :return: dataset length.
        """
        return self._len

    def __getitem__(self, index: int) -> dict:
        """
        Returning dataset element at input index.
        :param index: integer dataset index.
        :return: dictionary of relevant information.
        """

        if self.embedding_pre_loading:
            key = str(index)
            with h5py.File(self.embeddings_path, "r") as emb_h5, h5py.File(
                self.labels_path, "r"
            ) as lab_h5:
                self.emb = emb_h5[key][()]
                self.label = lab_h5[key][()]

            return {
                "emb": self.emb,
                "label": self.label,
            }

        # Getting path to label filenames
        if self.task_type == "linear_probing" or self.dataset_name == "pannuke":
            label = self.labels[index]
        elif self.task_type == "segmentation":
            (
                label_path,
                label_patch_i_min,
                label_patch_i_max,
                label_patch_j_min,
                label_patch_j_max,
            ) = self.labels[index]

            # Loading mask array and transforming to torch tensor
            label = Image.open(
                os.path.join(self.base_data_folder, self.dataset_name, label_path)
            )
            # Sampling patch in WSI
            label = F.pil_to_tensor(label)
            label = label[
                0,
                label_patch_i_min:label_patch_i_max,
                label_patch_j_min:label_patch_j_max,
            ].long()

            if self.dataset_name == "ocelot":
                # Re-mapping
                label[label == 1] = 0
                label[label == 2] = 1
                label[label == 255] = -1
        else:
            raise ValueError(f"{task_type} is not a valid task type")

        if not self.image_pre_loading:
            # Loading image
            image, viz_image = self.image_loading(index)
        else:
            image, viz_image = self.images[index]

        if viz_image is not None:
            return {
                "image": image,
                "label": label,
                "viz_image": viz_image,
            }
        else:
            return {
                "image": image,
                "label": label,
            }

    def image_loading(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loading an image at a given index and applying transforms.

        :param index: image index.
        :return: transformed image, viz image.
        """
        # Getting path to image filename
        if self.task_type == "linear_probing":
            image_path = self.images[index]
        elif self.task_type == "segmentation" and self.dataset_name != "pannuke":
            (
                image_path,
                im_patch_i_min,
                im_patch_i_max,
                im_patch_j_min,
                im_patch_j_max,
            ) = self.images[index]

        # Loading image
        if self.dataset_name == "patch_camelyon":
            image = Image.fromarray(image_path)
        elif self.dataset_name == "pannuke":
            image = self.images[index]
        else:
            image = Image.open(
                os.path.join(self.base_data_folder, self.dataset_name, image_path)
            ).convert("RGB")

        if self.task_type == "segmentation":
            if self.dataset_name != "pannuke":
                # Sampling patch in WSI
                image = F.pil_to_tensor(image)
                image = image[
                    :, im_patch_i_min:im_patch_i_max, im_patch_j_min:im_patch_j_max
                ]
            else:
                image = (
                    torch.Tensor(image.astype(np.uint8))
                    .to(torch.uint8)
                    .permute((2, 0, 1))
                )
            viz_image = image.clone()
            image = F.to_pil_image(image)
        else:
            viz_image = None

        # Applying model-specific transform
        image = self.transform(image)

        return image, viz_image
