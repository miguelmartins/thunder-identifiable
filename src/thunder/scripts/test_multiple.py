from typing import Tuple
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import lightning as L
import h5py

from src.thunder.utils.data import load_embeddings
import torchmetrics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        linear_projector: torch.nn.Linear,
        learning_rate: float,
        metrics: torchmetrics.MetricCollection,
    ) -> None:
        super().__init__()
        self.linear_projector = linear_projector
        self.learning_rate = learning_rate
        self.automatic_optimization = False
        self.train_metrics = metrics
        self.validation_metrics = self.train_metrics.clone(prefix="val_")
        print(f"num classes {linear_projector.weight.data.shape[0]}")

    def forward(self, batch):
        x, _ = batch
        return self.linear_projector(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear_projector(x)
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
        y_hat = self.linear_projector(x)
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


N = 10000
MAX_EPOCHS = 50
WEIGHT_DECAY = 0.05
LR = 1e-4
WARMUP = int(0.1 * MAX_EPOCHS)
N_RUNS = 100
NAMES = ["conch", "uni", "uni2h", "clipvitbasepatch32"]
EMBED = [512, 1024, 1536, 512]


def fit(name, embed):
    folder = f"/Users/miguelmartins/Projects/thunder-identifiable/src/thunder/embeddings/{name}"
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )[0]
    emb, labels = load_embeddings(embeddings_folder=folder, splits=["train"])
    for x, y in emb.items():
        embeddings = torch.from_numpy(y)
        emb[x] = EmbeddingsInstanceDataset(embeddings)
        print(type(x), type(y), type(embeddings), type(emb[x]))

    train = emb["train"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for run_number in range(N_RUNS):
        perm = torch.randperm(N)  # draw once
        train_samples = torch.utils.data.Subset(train, perm)
        assert len(train_samples) == N
        train_dl = torch.utils.data.DataLoader(
            train_samples, shuffle=True, batch_size=32
        )

        linear_map = torch.nn.Linear(
            in_features=embed, out_features=len(train_samples), bias=False
        )
        metrics = torchmetrics.MetricCollection(
            {
                "id_acc": torchmetrics.Accuracy(
                    task="multiclass",
                    # W projects from d to |I|, so we can use this to get the number of "instances"
                    num_classes=N,
                )
            }
        )

        id_model = InstanceLearner(
            linear_projector=linear_map, metrics=metrics, learning_rate=LR
        )
        logger = TensorBoardLogger(
            "logs", name=f"{name}", version=timestamp, sub_dir=str(run_number)
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
    run_all_threads()
