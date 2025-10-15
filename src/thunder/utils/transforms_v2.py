from __future__ import annotations

import math
from typing import Dict, Callable, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode
from PIL import Image

try:
    import kornia.morphology as _kmorph
except ModuleNotFoundError:
    _kmorph = None


# ---------------------------
# Seeding
# ---------------------------

_AUG_SEED: Optional[int] = None


def set_transform_seed(seed: int) -> None:
    """
    Seed augmentation RNGs used by v2 transforms and our wrappers.
    """
    global _AUG_SEED
    _AUG_SEED = seed
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_transform_seed() -> Optional[int]:
    return _AUG_SEED


# ---------------------------
# Small utilities (v2 prefers Tensor images; ensure types are OK)
# ---------------------------

# For transforms that need tensor domain (RandomErasing, Kornia ops),
# we’ll include ToImage/ToDtype in the pipelines where needed.


# ---------------------------
# Custom v2-compatible transforms
# ---------------------------


class RandomGamma(v2.Transform):
    """Sample gamma in [1+lo, 1+hi] and apply."""

    def __init__(self, gamma_add_range=(-0.5, 0.5)):
        super().__init__()
        self.lo, self.hi = gamma_add_range

    def _get_params(self, *args, **kwargs):
        g = 1.0 + torch.empty(1).uniform_(self.lo, self.hi).item()
        return {"gamma": g}

    def _transform(self, inpt, params):
        return v2.functional.adjust_gamma(inpt, gamma=params["gamma"], gain=1.0)


class HEDShift(v2.Transform):
    """
    HED-shift augmentation per your implementation, wrapped as v2.Transform.
    Works on Tensor or PIL; returns same type as input.
    """

    def __init__(self, sigma: float = 0.025):
        super().__init__()
        self.sigma = float(sigma)
        # Stain matrix and its inverse
        M = torch.tensor(
            [[0.651, 0.701, 0.290], [0.269, 0.568, 0.778], [0.633, -0.713, 0.302]],
            dtype=torch.float32,
        )
        self.register_buffer("_M", M)
        self.register_buffer("_RGB2HED", torch.linalg.inv(M))

    def _get_params(self, *args, **kwargs):
        alpha = torch.normal(mean=1.0, std=self.sigma, size=(1, 3))
        beta = torch.normal(mean=0.0, std=self.sigma, size=(1, 3))
        return {"alpha": alpha, "beta": beta}

    def _transform(self, inpt, params):
        # Convert to tensor [C,H,W], float in [0,1]
        img_t, was_pil = self._to_tensor(inpt)

        eps = 3.14159
        C, H, W = img_t.shape
        P = img_t.reshape(C, -1).movedim(0, -1)  # [N,3]

        S = torch.matmul(-torch.log(P + eps), self._RGB2HED)  # [N,3]
        Shat = params["alpha"] * S + params["beta"]
        Phat = torch.exp(-torch.matmul(Shat, self._M)) - eps
        Phat = torch.clamp(Phat, 0.0, 1.0)

        out = Phat.movedim(-1, 0).reshape(C, H, W)
        return self._from_tensor(out, was_pil)

    @staticmethod
    def _to_tensor(x):
        if isinstance(x, Image.Image):
            return v2.functional.to_image(x), True
        if torch.is_tensor(x):
            # assume already Image(TensorLike)
            t = x
            if t.dtype == torch.uint8:
                t = t.float().div(255)
            return t, False
        raise TypeError("HEDShift expects PIL.Image or Tensor image.")

    @staticmethod
    def _from_tensor(t, was_pil):
        if was_pil:
            return v2.functional.to_image_pil(t)
        return t


class KorniaMorph(v2.Transform):
    """
    Wrap kornia morphology ops as v2 transforms.
    op_name in {"dilation", "erosion", "opening", "closing"}.
    """

    def __init__(self, op_name: str, max_kernel: int = 5):
        super().__init__()
        if _kmorph is None:
            raise ImportError(
                "kornia is required for morphology transforms; `pip install kornia`."
            )
        self.op_name = op_name
        self.max_kernel = int(max_kernel)

    def _get_params(self, *args, **kwargs):
        k = int(torch.randint(2, self.max_kernel + 1, (1,)).item())
        if k % 2 == 0:
            k += 1
        return {"k": k}

    def _transform(self, inpt, params):
        # Ensure tensor float [0,1]
        t = v2.functional.to_image(inpt)
        if t.dtype == torch.uint8:
            t = t.float().div(255)

        k = params["k"]
        kernel = torch.ones((k, k), device=t.device)
        x = t.unsqueeze(0)  # [1,C,H,W] expected by kornia

        if self.op_name == "dilation":
            y = _kmorph.dilation(x, kernel)
        elif self.op_name == "erosion":
            y = _kmorph.erosion(x, kernel)
        elif self.op_name == "opening":
            y = _kmorph.opening(x, kernel)
        elif self.op_name == "closing":
            y = _kmorph.closing(x, kernel)
        else:
            raise ValueError(f"Unknown op_name {self.op_name}")

        out = y.squeeze(0)
        # Convert back to match input type (keep as tensor; caller can ToPIL if needed)
        return out


class RandomFiveCrop(v2.Transform):
    """
    Pick one of the 5 standard crops (TL, TR, BL, BR, center) of size = min(H,W)//2.
    """

    def __init__(self):
        super().__init__()

    def _get_params(self, inpt):
        # We need size based on image dims; infer after converting to PIL or Tensor
        if isinstance(inpt, Image.Image):
            w, h = inpt.size
        else:
            t = v2.functional.to_image(inpt)
            _, h, w = t.shape
        size = min(w, h) // 2
        idx = int(torch.randint(0, 5, (1,)).item())
        return {"size": size, "idx": idx}

    def _transform(self, inpt, params):
        crops = v2.FiveCrop(size=params["size"])(inpt)
        return crops[params["idx"]]


# ---------------------------
# Factory
# ---------------------------


def get_invariance_transforms_v2() -> Dict[str, Callable]:
    """
    Return a dict of name -> v2-compatible transform (or v2.Compose).
    Each transform returns an augmented image.
    """
    return {
        # Either horizontal or vertical flip with equal chance
        "random_flip": v2.RandomChoice(
            [
                v2.RandomHorizontalFlip(p=1.0),
                v2.RandomVerticalFlip(p=1.0),
            ]
        ),
        # Random 90/180/270 rotation
        "random_rotate": v2.RandomChoice(
            [
                v2.RandomRotation(
                    90, interpolation=InterpolationMode.BILINEAR, expand=True
                ),
                v2.RandomRotation(
                    180, interpolation=InterpolationMode.BILINEAR, expand=True
                ),
                v2.RandomRotation(
                    270, interpolation=InterpolationMode.BILINEAR, expand=True
                ),
            ]
        ),
        # Translate/scale/shear in one affine
        "random_translate": v2.RandomAffine(
            degrees=0.0,
            translate=(0.20, 0.20),  # max_frac
            scale=(1.0 - 0.20, 1.0 + 0.20),
            shear=(-0.20 * 10 / 2, 0.20 * 10 / 2),
            interpolation=InterpolationMode.BILINEAR,
        ),
        "random_gaussian_blur": v2.GaussianBlur(kernel_size=15),
        "random_color_jitter": v2.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.35
        ),
        "random_gamma": RandomGamma(gamma_add_range=(-0.5, 0.5)),
        "random_hed": HEDShift(sigma=0.025),
        # Cutout -> RandomErasing (square; erase with zeros)
        "random_cutout": v2.Compose(
            [
                to_tensor_01,
                v2.RandomErasing(
                    p=1.0, scale=(0.10, 0.50), ratio=(1.0, 1.0), value=0.0
                ),
            ]
        ),
        # Morphology ops (require kornia)
        "random_dilation": KorniaMorph("dilation", max_kernel=5),
        "random_erosion": KorniaMorph("erosion", max_kernel=5),
        "random_opening": KorniaMorph("opening", max_kernel=5),
        "random_closing": KorniaMorph("closing", max_kernel=5),
        "five_crop": RandomFiveCrop(),
        "elastic_transform": v2.ElasticTransform(alpha=250.0, sigma=6.0),
    }


__all__ = ["get_invariance_transforms_v2"]
