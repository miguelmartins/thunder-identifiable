from typing import Tuple
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import lightning as L
import h5py

from src.thunder.utils.data import load_embeddings
from src.thunder.models.pretrained_models import load_pretrained_model
import torchmetrics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from src.thunder.utils.config import get_config
from src.thunder.utils.utils import print_task_hyperparams
from src.thunder.models.pretrained_models import get_model_from_name
from src.thunder.utils.data import get_data
from src.thunder.utils.data import PatchDataset
import itertools
from torchvision import transforms
from src.thunder.tasks.transformation_invariance import (
    _pil_from_any,
    _compute_batch_embeddings,
)

N = 10000
MAX_EPOCHS = 50
WEIGHT_DECAY = 0.05
LR = 1e-4
WARMUP = int(0.1 * MAX_EPOCHS)
N_RUNS = 3
MODELS = ["conch", "uni", "clipvitbasepatch32"]
EMBED_ = [512, 1024, 1536, 512]
EMBED = {name: emb for name, emb in zip(MODELS, EMBED_)}
DATASETS = [
    "bach",
    "bracs",
    "break_his",
    "ccrcc",
    "crc",
    "esca",
    # "mhist",
    "ocelot",
    "pannuke",
    "patch_camelyon",
    # "segpath_epithelial",
    # "segpath_lymphocytes",
    "tcga_crc_msi",
    "tcga_tils",
    "tcga_uniform",
    # "wilds",
]

DEV_DATASETS = [
    "crc",  # 40x Breast 52k
    "esca",  # 10x Oeso 367k
    "tcga_tils",  # 20x Multi 304k
]


class EmbeddingsInstanceDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        return self.embeddings[index], index

    def __len__(self) -> int:
        return len(self.embeddings)


class InstanceLearner(L.LightningModule):
    def __init__(
        self,
        f: torch.nn.Module,
        linear_projector: torch.nn.Linear,
        learning_rate: float,
        metrics: torchmetrics.MetricCollection,
        device: str,
    ) -> None:
        super().__init__()
        self.f = f
        self.linear_projector = linear_projector
        self.learning_rate = learning_rate
        self.automatic_optimization = False
        self.train_metrics = metrics
        self.validation_metrics = self.train_metrics.clone(prefix="val_")
        print(f"num classes {linear_projector.weight.data.shape[0]}")

    def _process_batch(self, batch):
        if isinstance(batch, dict):
            batch = batch["image"].to(self.device)
            # batch = torch.stack([_pil_from_any(img) for img in batch]).to(self.device)
        # Compute embeddings for original images
        return batch

    def forward(self, batch):
        print(type(batch))
        x, _ = batch
        x = self._process_batch(x)
        return self.linear_projector(self.f(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self._process_batch(x)
        y_hat = self.linear_projector(self.f(x))
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        metric_values = self.train_metrics(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(metric_values, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        try:
            for scheduler in schedulers:
                scheduler.step()
        except TypeError:
            schedulers.step()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self._process_batch(x)
        y_hat = self.linear_projector(self.f(x))
        # - Input: Shape `(C)`, `(N, C)` or `(N, C, d_1, d_2, ..., d_K)` with `K \geq 1`
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        metric_values = self.validation_metrics(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(metric_values, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=1e-8, end_factor=1, total_iters=WARMUP
        )
        cosine_ann = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS
        )

        return [optimizer], [warmup, cosine_ann]  # [warmup, cosine_ann]


# load_pretrained_model(cfg: DictConfig, adaptation_type: str, device: str)
def simclr_augmentations(image_size, jitter_strength):
    # SimCLR uses: B/C/S = 0.8*s, Hue = 0.2*s ; applied with prob 0.8
    color_jitter = transforms.ColorJitter(
        brightness=0.8 * jitter_strength,
        contrast=0.8 * jitter_strength,
        saturation=0.8 * jitter_strength,
        hue=0.2 * jitter_strength,
    )

    # Gaussian blur kernel size ~ 0.1 * image_size, must be odd and >= 3
    k = max(3, int(round(0.1 * image_size)))
    if k % 2 == 0:
        k += 1

    augmentation = [
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))], p=0.5
        ),
    ]
    return transforms.Compose(augmentation)


def fit(model_name, embed, dataset_name="tcga_crc_msi", id_n=10**4):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )[0]
    model, preprocessing, get_embeddings = get_model_from_name(
        model_name, device="cuda"
    )
    model.eval()
    model = model.to(device)
    f_ = lambda x: get_embeddings(x, model, "linear_probing")
    model.eval()
    base_data_folder = os.path.join(os.environ["THUNDER_BASE_DATA_FOLDER"], "datasets")
    print(model_name, base_data_folder, dataset_name)
    data = get_data(dataset_name=dataset, base_data_folder=base_data_folder)
    simclr_augs = simclr_augmentations(
        image_size=224,  # TODO: change this later dep. on dataset
        jitter_strength=1.0,
    )
    transform_ = transforms.Compose([simclr_augs, preprocessing])
    train_ = PatchDataset(
        data["train"]["images"],
        data["train"]["labels"],
        transform=transform_,  # transformations handled manually later
        task_type="linear_probing",
        dataset_name=dataset_name,
        base_data_folder=base_data_folder,
        embeddings_folder=None,
        image_pre_loading=False,
        embedding_pre_loading=False,
    )
    train = EmbeddingsInstanceDataset(train_)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    id_n = min(id_n, len(train))
    for run_number in range(N_RUNS):
        perm = torch.randperm(id_n)  # draw once
        train_samples = torch.utils.data.Subset(train, perm)
        train_dl = torch.utils.data.DataLoader(
            train_samples,
            shuffle=True,
            batch_size=32,
            num_workers=20,
        )

        linear_map = torch.nn.Linear(
            in_features=embed, out_features=len(train_samples), bias=False
        )
        linear_map = linear_map.to(device)
        metrics = torchmetrics.MetricCollection(
            {
                "id_acc": torchmetrics.Accuracy(
                    task="multiclass",
                    # W projects from d to |I|, so we can use this to get the number of "instances"
                    num_classes=id_n,
                )
            }
        )

        id_model = InstanceLearner(
            f=f_,
            linear_projector=linear_map,
            metrics=metrics,
            learning_rate=LR,
            device=device,
        )
        logger = TensorBoardLogger(
            "logs", name=f"{model_name}", version=timestamp, sub_dir=str(run_number)
        )
        trainer = L.Trainer(
            max_epochs=MAX_EPOCHS,
            devices=1,
            accelerator=device,
            logger=logger,
            reload_dataloaders_every_n_epochs=1,  # make sure we create a new permutation
        )

        trainer.fit(id_model, train_dl)


def run_all_threads():
    with ThreadPoolExecutor(max_workers=len(NAMES)) as pool:
        futures = {
            pool.submit(fit, name, embed_dim): name
            for name, embed_dim in zip(NAMES, EMBED)
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                print(f"[OK] {name} finished.")
            except Exception as e:
                print(f"[ERR] {name} failed: {e}")


if __name__ == "__main__":
    models = os.listdir(
        os.path.join(os.environ["THUNDER_BASE_DATA_FOLDER"], "pretrained_ckpts")
    )

    for dataset, model in itertools.product(DEV_DATASETS, MODELS):
        fit(model_name=model, embed=EMBED[model], dataset_name=dataset)
    # config = get_config(**config_dict)
    # print_task_hyperparams(config)
    # run_all_threads()
