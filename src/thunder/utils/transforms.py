from __future__ import annotations

import random
from typing import Callable, Dict, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as v2
from PIL import Image
from torchvision.transforms import ColorJitter, InterpolationMode

try:
    import kornia.morphology as _kmorph
except ModuleNotFoundError:
    _kmorph = None

_Image = Union[Image.Image, torch.Tensor, np.ndarray]


_AUG_RNG = random.Random()


def set_transform_seed(seed: int) -> None:
    """
    Seed only the augmentation RNG.  Call this once per‐dataset before you apply any of the get_invariance_transforms().
    :param seed: an integer seed value.
    """
    _AUG_RNG.seed(seed)
    global _AUG_SEED
    _AUG_SEED = seed


def get_transform_seed() -> int | None:
    """
    Get the seed (for debugging)
    """
    return _AUG_SEED


def _to_tensor(img: _Image) -> torch.Tensor:
    """
    Converting image to float tensor in [0, 1] while preserving contents.

    :param img: input image (PIL, Tensor or ndarray).
    :return: tensor representation of image.
    """
    if isinstance(img, torch.Tensor):
        t = img.clone()
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        return t
    if isinstance(img, Image.Image):
        return F.to_tensor(img)
    arr = img
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    t = torch.from_numpy(arr)
    if t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)
    return t


def _from_tensor(t: torch.Tensor, ref: _Image) -> _Image:
    """
    Returning tensor in the same container type as reference image (PIL, Tensor or ndarray).

    :param t: tensor to convert.
    :param ref: reference image for output type.
    :return: output image.
    """
    t_clamped = t.clamp(0.0, 1.0)
    if isinstance(ref, torch.Tensor):
        if ref.dtype == torch.uint8:
            return (t_clamped * 255).to(torch.uint8)
        return t_clamped
    if isinstance(ref, Image.Image):
        return F.to_pil_image(t_clamped)
    arr = t_clamped.permute(1, 2, 0).cpu().numpy()
    if ref.dtype == np.uint8:
        return (arr * 255).astype(np.uint8)
    return arr


def _identity(img: _Image) -> Tuple[_Image, _ParamDict]:
    """
    Identity transform (no change).

    :param img: input image.
    :return: same input image.
    """
    return img, {}


def _random_flip(p: float = 0.5) -> Callable[[_Image], _Image]:
    """
    Randomly flip image horizontally or vertically with probability p.

    :param p: flip probability.
    :return: flip transform.
    """

    def _inner(img: _Image):
        horizontal = _AUG_RNG.random() < p
        out = F.hflip(img) if horizontal else F.vflip(img)
        return out, {"orientation": "horizontal" if horizontal else "vertical"}

    return _inner


def _random_rotate() -> Callable[[_Image], _Image]:
    """
    Random rotation by 90°, 180°, or 270°.

    :return: rotation transform.
    """

    def _inner(img: _Image):
        angle = _AUG_RNG.randint(1, 3) * 90
        return (
            F.rotate(img, angle, InterpolationMode.BILINEAR, expand=True),
            {"angle": angle},
        )

    return _inner


def _random_translate(max_frac: float = 0.20) -> Callable[[_Image], _Image]:
    """
    Random translation, scale, and shear transform.

    :param max_frac: maximum fraction for translation and scaling.
    :return: affine transform.
    """

    def _inner(img: _Image):
        if isinstance(img, Image.Image):
            w, h = img.size
        elif isinstance(img, torch.Tensor):
            _, h, w = img.shape
        else:
            h, w = img.shape[:2]
        dx = int(_AUG_RNG.uniform(-max_frac, max_frac) * w)
        dy = int(_AUG_RNG.uniform(-max_frac, max_frac) * h)
        scale = 1.0 + _AUG_RNG.uniform(-max_frac, max_frac)
        shear = _AUG_RNG.uniform(-max_frac * 10 / 2, max_frac * 10 / 2)
        out = F.affine(
            img,
            angle=0.0,
            translate=(dx, dy),
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.BILINEAR,
        )
        return out, {"dx": dx, "dy": dy, "scale": scale, "shear": shear}

    return _inner


def _random_gaussian_blur(kernel_size: int = 15) -> Callable[[_Image], _Image]:
    """
    Applying Gaussian blur with fixed kernel size.

    :param kernel_size: size of the Gaussian kernel.
    :return: blur transform.
    """

    def _inner(img: _Image):
        return F.gaussian_blur(img, kernel_size=kernel_size), {
            "kernel_size": kernel_size
        }

    return _inner


def _random_color_jitter(
    brightness: float = 0.5,
    contrast: float = 0.5,
    saturation: float = 0.5,
    hue: float = 0.35,
) -> Callable[[_Image], _Image]:
    """
    Randomly adjusts brightness, contrast, saturation, and hue within specified max ranges.

    :param brightness: Max brightness jitter factor.
    :param contrast: Max contrast jitter factor.
    :param saturation: Max saturation jitter factor.
    :param hue: Max hue jitter factor
    :return: Color jitter transform.
    """

    def _sample_factor(max_delta: float) -> float:
        return _AUG_RNG.uniform(max(0, 1.0 - max_delta), 1.0 + max_delta)

    def _inner(img: _Image):
        bf = _sample_factor(brightness)
        cf = _sample_factor(contrast)
        sf = _sample_factor(saturation)
        hf = _AUG_RNG.uniform(-hue, hue)

        out = F.adjust_brightness(img, bf)
        out = F.adjust_contrast(out, cf)
        out = F.adjust_saturation(out, sf)
        out = F.adjust_hue(out, hf)

        return out, {
            "brightness_factor": bf,
            "contrast_factor": cf,
            "saturation_factor": sf,
            "hue_factor": hf,
        }

    return _inner


def _random_gamma(
    gamma_range: Tuple[float, float] = (-0.5, 0.5)
) -> Callable[[_Image], _Image]:
    """
    Random gamma adjustment: gamma is sampled in [1 + min, 1 + max] = [0.5, 1.5].

    :param gamma_range: range for additive gamma sampling (e.g., [-0.5, 0.5]).
    :return: gamma adjustment transform.
    """

    def _inner(img: _Image):
        gamma = 1 + _AUG_RNG.uniform(*gamma_range)
        return F.adjust_gamma(img, gamma=gamma, gain=1.0), {"gamma": gamma}

    return _inner


def _random_hed(sigma: float = 0.025) -> Callable[[_Image], _Image]:
    """
    Applying HED-shift augmentation to an image.

    HED augmentation method as described in:
    [1] Faryna, K., Van der Laak, J., Litjens, G., 2021. Tailoring automated data augmentation to H&E-stained histopathology.
    In Medical Imaging with Deep Learning.

    :param sigma: standard deviation for perturbations in HED space.
    :return: HED augmentation transform.
    """

    def _inner(img: _Image):
        M = torch.tensor(
            np.array(
                [[0.651, 0.701, 0.290], [0.269, 0.568, 0.778], [0.633, -0.713, 0.302]],
                dtype="float32",
            )
        )
        RGB2HED = torch.linalg.inv(M)

        # Handle input type
        if isinstance(img, Image.Image):
            # Convert PIL Image to tensor
            img = F.to_tensor(img)
            is_pil = True  # Flag to check if input was a PIL image
        elif isinstance(img, torch.Tensor):
            if img.dim() != 3:
                raise ValueError("Input tensor must have shape C x H x W.")
            is_pil = False
        else:
            raise TypeError("Input must be a PIL Image or a PyTorch tensor.")

        epsilon = 3.14159
        C, X, Y = img.shape  # Remove batch dimension

        # Reshape image P \in R^(N,3)
        P = img.reshape(C, -1).movedim(0, -1)  # Move channel to the last dimension

        # HED images
        S = torch.matmul(-torch.log(P + epsilon), RGB2HED)

        # Channel-wise perturbations
        alpha = torch.normal(mean=1, std=sigma, size=[1, 3])  # Change B to 1
        beta = torch.normal(mean=0, std=sigma, size=[1, 3])  # Change B to 1
        Shat = alpha * S + beta

        # Augmented RGB images
        Phat = torch.exp(-torch.matmul(Shat, M)) - epsilon

        # Clip values to range [0, 255]
        Phat_clipped = torch.clip(Phat, min=0.0, max=1.0)

        out = Phat_clipped.movedim(-1, 0).reshape(
            C, X, Y
        )  # Move channel back to the first dimension

        out = F.to_pil_image(out) if is_pil else out

        # Return the output in the same format as input
        return out, {
            "alpha": alpha.squeeze().tolist(),
            "beta": beta.squeeze().tolist(),
            "sigma": sigma,
        }

    return _inner


def _random_cutout(max_mask_frac: float = 0.50) -> Callable[[_Image], _Image]:
    """
    Applying a random square cutout covering up to max fraction of min(H, W).

    :param max_mask_frac: maximum mask size fraction.
    :return: cutout transform.
    """

    def _inner(img: _Image):
        tensor = _to_tensor(img)
        _, h, w = tensor.shape
        size = int(_AUG_RNG.uniform(0.10, max_mask_frac) * min(h, w))
        top = _AUG_RNG.randint(0, h - size)
        left = _AUG_RNG.randint(0, w - size)
        tensor[:, top : top + size, left : left + size] = 0.0
        out = _from_tensor(tensor, img)
        return out, {
            "size": size,
            "top": top,
            "left": left,
            "mask_frac": max_mask_frac,
        }

    return _inner


def _random_dilation(max_kernel: int = 5) -> Callable[[_Image], _Image]:
    """
    Random dilation with square kernel of random size.

    :param max_kernel: maximum kernel size.
    :return: dilation transform.
    """
    if _kmorph is None:
        raise ImportError(
            "kornia is required for the dilation transform; install via `pip install kornia`."
        )

    def _inner(img: _Image):
        t = _to_tensor(img).unsqueeze(0)
        k = _AUG_RNG.randint(2, max_kernel)
        if k % 2 == 0:
            k += 1
        kernel = torch.ones((k, k), device=t.device)
        out = _kmorph.dilation(t, kernel).squeeze(0)
        return _from_tensor(out, img), {"kernel_size": k}

    return _inner


def _random_erosion(max_kernel: int = 5) -> Callable[[_Image], _Image]:
    """
    Random erosion with square kernel of random size.

    :param max_kernel: maximum kernel size.
    :return: erosion transform.
    """
    if _kmorph is None:
        raise ImportError(
            "kornia is required for the erosion transform; install via `pip install kornia`."
        )

    def _inner(img: _Image):
        t = _to_tensor(img).unsqueeze(0)
        k = _AUG_RNG.randint(2, max_kernel)
        if k % 2 == 0:
            k += 1
        kernel = torch.ones((k, k), device=t.device)
        out = _kmorph.erosion(t, kernel).squeeze(0)
        return _from_tensor(out, img), {"kernel_size": k}

    return _inner


def _random_opening(max_kernel: int = 5) -> Callable[[_Image], _Image]:
    """
    Random opening with square kernel of random size.

    :param max_kernel: maximum kernel size.
    :return: opening transform.
    """
    if _kmorph is None:
        raise ImportError(
            "kornia is required for the opening transform; install via `pip install kornia`."
        )

    def _inner(img: _Image):
        t = _to_tensor(img).unsqueeze(0)
        k = _AUG_RNG.randint(2, max_kernel)
        if k % 2 == 0:
            k += 1
        kernel = torch.ones((k, k), device=t.device)
        out = _kmorph.opening(t, kernel).squeeze(0)
        return _from_tensor(out, img), {"kernel_size": k}

    return _inner


def _five_crop_random():
    """
    Choose one of torchvision’s 5 standard crops (4 corners + center).

    Returns (cropped_img, {"crop_id": 1‑5, "size": size})
    """

    def _inner(img):
        # 5‑tuple of (TL, TR, BL, BR, center)
        width, height = img.size
        size = min(width, height) // 2
        crops = F.five_crop(img, size)
        crop_id = _AUG_RNG.randint(0, 4)  # 1‑based like your example
        out = crops[crop_id]  # replicate `img = img[param-1]`
        return out, {"crop_id": crop_id, "size": size}

    return _inner


def _random_closing(max_kernel: int = 5) -> Callable[[_Image], _Image]:
    """
    Random closing with square kernel of random size.

    :param max_kernel: maximum kernel size.
    :return: closing transform.
    """
    if _kmorph is None:
        raise ImportError(
            "kornia is required for the closing transform; install via `pip install kornia`."
        )

    def _inner(img: _Image):
        t = _to_tensor(img).unsqueeze(0)
        k = _AUG_RNG.randint(2, max_kernel)
        if k % 2 == 0:
            k += 1
        kernel = torch.ones((k, k), device=t.device)
        out = _kmorph.closing(t, kernel).squeeze(0)
        return _from_tensor(out, img), {"kernel_size": k}

    return _inner


def _elastic_transform(alpha: float = 250.0, sigma: float = 6.0):
    """
    Wraps torchvision.v2.ElasticTransform so it fits (img)->(img,params).
    alpha: magnitude; sigma: Gaussian smoothing of displacement field.
    """
    transform = v2.ElasticTransform(alpha=alpha, sigma=sigma)

    def _inner(img):
        out = transform(img)
        return out, {"alpha": alpha, "sigma": sigma}

    return _inner


def get_invariance_transforms() -> Dict[str, Callable[[_Image], _Image]]:
    """
    Getting dictionary of available data augmentation transformations.

    :return: dictionary {name: transform callable}.
    """
    return {
        "identity": _identity,
        "random_flip": _random_flip(),
        "random_rotate": _random_rotate(),
        "random_translate": _random_translate(),
        "random_gaussian_blur": _random_gaussian_blur(),
        "random_color_jitter": _random_color_jitter(),
        "random_gamma": _random_gamma(),
        "random_hed": _random_hed(),
        "random_cutout": _random_cutout(),
        "random_dilation": _random_dilation(),
        "random_erosion": _random_erosion(),
        "random_opening": _random_opening(),
        "random_closing": _random_closing(),
        "five_crop": _five_crop_random(),
        "elastic_transform": _elastic_transform(),
    }


__all__ = ["get_invariance_transforms"]
