from __future__ import annotations

import random
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


# Denormalize
def denormalize(tensor):
    return tensor * std + mean


NormType = Literal["linf", "l2"]

__all__ = ["PGDImageAttack"]


class PGDImageAttack(nn.Module):
    """Projected Gradient Descent (PGD) adversarial attack for image *classification*.

    The attack perturbs input images so that a fixed feature extractor ``fm_model``
    together with a pre‑trained linear classifier ``linear`` mis‑classifies them.
    Supports both untargeted (default) and targeted variants and two norm
    constraints (``linf`` and ``l2``).
    """

    def __init__(
        self,
        fm_model: nn.Module,
        linear: nn.Module,
        extract_embedding: Callable[[torch.Tensor, nn.Module, str], torch.Tensor],
        *,
        eps: float,
        alpha: float,
        num_steps: int,
        norm: NormType = "linf",
        random_start: bool = True,
        targeted: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()

        if norm not in {"linf", "l2"}:
            raise ValueError("norm must be 'linf' or 'l2'")

        self.fm_model = fm_model.eval().to(device)
        self.linear = linear.eval().to(device)

        self.eps = eps
        self.alpha = alpha
        self.num_steps = num_steps
        self.norm = norm
        self.random_start = random_start
        self.targeted = targeted
        self.device = torch.device(device)
        self.extract_embedding = extract_embedding

        # Small constant to avoid division by zero
        self._tiny = 1e-12

    # ---------------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------------
    def perturb(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate adversarial examples.

        Args:
            images: `(B, C, H, W)` tensor in **input space** (already normalised
                as expected by ``fm_model`` / ``linear``).
            labels: ground‑truth class indices `(B,)`.
        Returns:
            Tuple ``(adv_images, adv_logits)`` where ``adv_images`` are the
            perturbed inputs and ``adv_logits`` are the classifier outputs on
            those inputs.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            clean_logits = self._forward_classifier(images)

        self.img_min = images.amin(dim=(2, 3), keepdim=True)
        self.img_max = images.amax(dim=(2, 3), keepdim=True)

        # Initialise perturbation
        adv = self._init_adv(images)

        for _ in range(self.num_steps):
            adv = self._pgd_step(images, adv, labels)

        # Final logits for convenience
        with torch.no_grad():
            adv_logits = self._forward_classifier(adv)

        return clean_logits.detach(), adv_logits.detach()

    def _forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and linear head."""
        feat = self.extract_embedding(x, self.fm_model, "linear_probing")
        logits = self.linear(feat)
        return logits

    def _init_adv(self, images: torch.Tensor) -> torch.Tensor:
        """Return initial adversarial tensor."""
        if not self.random_start:
            return images.clone().detach()

        delta = torch.empty_like(images).uniform_(-self.eps, self.eps)
        adv = images + delta
        return adv.clamp(min=self.img_min, max=self.img_max).detach()

    def _pgd_step(
        self, x_orig: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        adv.requires_grad_(True)
        logits = self._forward_classifier(adv)
        loss = F.cross_entropy(logits, labels, reduction="mean")

        grad = torch.autograd.grad(loss, adv)[0]

        with torch.no_grad():
            # Normalise gradient
            grad = self._grad_sign_or_normalise(grad)
            adv = adv + self.alpha * (self.img_max - self.img_min) * grad

            # Project back into epsilon‑ball
            adv = self._project(x_orig, adv)
            return adv.clamp(min=self.img_min, max=self.img_max)

    def _grad_sign_or_normalise(self, grad: torch.Tensor) -> torch.Tensor:
        return grad.sign()

    def _project(self, x_orig: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        delta = x_adv - x_orig

        shift = self.img_max - self.img_min
        delta = delta.clamp(min=-self.eps * shift, max=self.eps * shift)
        return x_orig + delta
