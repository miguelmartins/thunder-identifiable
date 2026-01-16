from typing import Tuple
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import lightning as L
import h5py

from src.thunder.utils.data import load_embeddings
import torchmetrics
from datetime import datetime

import math


class SGLD(torch.optim.Optimizer):
    """
    Stochastic Gradient Langevin Dynamics (Welling & Teh, 2011)
    - weight_decay acts as Gaussian prior: grad += wd * p
    - step adds Gaussian noise with std = sqrt(2*lr)
    """

    def __init__(self, params, lr=1e-5, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            noise_std = math.sqrt(2.0 * lr)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)  # ∇(wd/2)||p||^2 = wd * p

                p.add_(grad, alpha=-lr)  # SGD step
                p.add_(torch.randn_like(p) * noise_std)  # Langevin noise
        return loss


class EmbeddingsInstanceDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        return self.embeddings[index], index

    def __len__(self) -> int:
        return len(self.embeddings)


@torch.no_grad()
def fit_feature_zscore(train_loader, device="cuda", eps=1e-6):
    """
    Computes per-dimension mean/std of encoder features on TRAIN data only.
    Assumes encoder is frozen (eval mode recommended).
    Returns: mean [d], std [d]
    """
    n_total = 0
    sum_vec = None
    sumsq_vec = None

    for z, _ in train_loader:
        if (
            z.dim() > 2
        ):  # if encoder outputs spatial maps, pool or flatten appropriately
            z = z.mean(dim=(2, 3))  # example: global average pool for CNNs

        # z = z.detach()
        B, d = z.shape
        if sum_vec is None:
            sum_vec = z.sum(dim=0)
            sumsq_vec = (z * z).sum(dim=0)
        else:
            sum_vec += z.sum(dim=0)
            sumsq_vec += (z * z).sum(dim=0)
        n_total += B

    mean = sum_vec / n_total
    var = sumsq_vec / n_total - mean * mean
    std = torch.sqrt(torch.clamp(var, min=eps))  # avoid zeros
    mean = mean.to(device)
    std = std.to(device)
    return mean, std


@torch.no_grad()
def zscore_normalize(z, mean, std):
    if z.dim() > 2:
        z = z.mean(dim=(2, 3))
    return (z - mean) / std


class InstanceLearnerLangevin(L.LightningModule):
    def __init__(
        self,
        linear_projector: torch.nn.Linear,
        learning_rate: float,
        sigma: float,
        mean: float,
        std: float,
        metrics: torchmetrics.MetricCollection,
    ) -> None:
        super().__init__()
        self.linear_projector = linear_projector
        self.learning_rate = learning_rate
        self.automatic_optimization = False
        self.sigma = sigma
        self.mean = mean
        self.std = std
        self.train_metrics = metrics
        self.validation_metrics = self.train_metrics.clone(prefix="val_")
        print(f"num classes {linear_projector.weight.data.shape[0]}")

    def forward(self, batch):
        x, _ = batch
        x = zscore_normalize(x, self.mean, self.std)
        return self.linear_projector(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.linear_projector.weight.data = (
            self.linear_projector.weight.data
            / torch.norm(self.linear_projector.weight.data, p=2, dim=-1, keepdim=True)
        )
        x = zscore_normalize(x, self.mean, self.std)
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
        if schedulers is not None:
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
        optimizer = SGLD(self.parameters(), lr=LR, weight_decay=1.0 / (self.sigma**2))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=1e-8, end_factor=1, total_iters=WARMUP
        )
        cosine_ann = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS
        )

        return optimizer  # , [warmup, cosine_ann]  # [warmup, cosine_ann]


N = 1000
MAX_EPOCHS = 1000
WEIGHT_DECAY = 0.05
LR = 1e-4
WARMUP = int(0.1 * MAX_EPOCHS)
N_RUNS = 1
SIGMA = 10


def main():
    folder = "/Users/miguelmartins/Projects/thunder-identifiable/src/thunder/embeddings/conch/"
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
        mean, std = fit_feature_zscore(train_dl, device=device)
        linear_map = torch.nn.Linear(
            in_features=512, out_features=len(train_samples), bias=False
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

        id_model = InstanceLearnerLangevin(
            linear_projector=linear_map,
            metrics=metrics,
            learning_rate=LR,
            sigma=SIGMA,
            mean=mean,
            std=std,
        )
        # logger = TensorBoardLogger(
        #     "logs", name="conch", version=timestamp, sub_dir=str(run_number)
        # )
        trainer = L.Trainer(
            max_epochs=MAX_EPOCHS,
            devices=1,
            accelerator=device,
            # logger=logger,
            reload_dataloaders_every_n_epochs=1,  # make sure we create a new permutation
        )

        trainer.fit(id_model, train_dl)
        preds = trainer.predict(id_model, train_dl)
        print(preds)


if __name__ == "__main__":
    main()
