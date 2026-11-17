"""Utilities for classifying real or generated images.

The classifier is intentionally kept separate from the DDPM: a diffusion
model's conditioning label is the requested class, not a prediction of the
class actually present in the generated image.

Example::

    from classifier import classify_image

    result = classify_image(
        model,
        "generated_samples/fish/sample_00000.png",
        class_names=dataset.classes,
        device="cuda",
    )
    print(result["label"], result["confidence"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision import transforms


DEFAULT_IMAGE_SIZE = 64
_NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def _device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return resolved


def _transform(image_size: int) -> transforms.Compose:
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        _NORMALIZE,
    ])


def _as_tensor(
    image: str | Path | Image.Image | torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Convert a path, PIL image, or image tensor to one normalized CHW tensor."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")

    if isinstance(image, Image.Image):
        return _transform(image_size)(image)
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a path, PIL image, or torch.Tensor")

    tensor = image.detach().float()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("classify_image expects one image, not a batch")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError("image tensor must have shape CHW or 1CHW")
    if tensor.shape[0] not in (1, 3):
        raise ValueError("image tensor must have one or three channels")
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    if tensor.min() < 0 or tensor.max() > 1:
        # Generated samples are [0, 1], while callers sometimes provide the
        # DDPM's normalized [-1, 1] representation.
        if tensor.min() >= -1 and tensor.max() <= 1:
            tensor = (tensor + 1) / 2
        else:
            raise ValueError("image tensor values must be in [0, 1] or [-1, 1]")
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False
    )[0]
    return _NORMALIZE(tensor)


@torch.inference_mode()
def classify_image(
    model: torch.nn.Module,
    image: str | Path | Image.Image | torch.Tensor,
    class_names: Sequence[str],
    *,
    device: str | torch.device = "cpu",
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, object]:
    """Classify one image and return its label, index, confidence, and scores.

    ``model`` must return one logit per class. ``class_names[index]`` must match
    the class ordering used to train that model (the alphabetical ordering from
    ``torchvision.datasets.ImageFolder`` is the usual choice).
    """
    if not class_names:
        raise ValueError("class_names must contain at least one class")
    target_device = _device(device)
    model = model.to(target_device)
    model.eval()
    inputs = _as_tensor(image, image_size).unsqueeze(0).to(target_device)
    logits = model(inputs)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(class_names):
        raise ValueError(
            f"model must return shape (1, {len(class_names)}), got {tuple(logits.shape)}"
        )
    probabilities = logits.softmax(dim=1)[0]
    index = int(probabilities.argmax())
    return {
        "index": index,
        "label": str(class_names[index]),
        "confidence": float(probabilities[index]),
        "scores": {str(name): float(probabilities[i]) for i, name in enumerate(class_names)},
    }


@torch.inference_mode()
def classify_images(
    model: torch.nn.Module,
    images: Sequence[str | Path | Image.Image | torch.Tensor],
    class_names: Sequence[str],
    *,
    device: str | torch.device = "cpu",
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = 64,
) -> list[dict[str, object]]:
    """Classify multiple images in batches using the same result format."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not images:
        return []
    target_device = _device(device)
    model = model.to(target_device)
    model.eval()
    tensors = [_as_tensor(image, image_size) for image in images]
    results: list[dict[str, object]] = []
    for start in range(0, len(tensors), batch_size):
        inputs = torch.stack(tensors[start:start + batch_size]).to(target_device)
        logits = model(inputs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim != 2 or logits.shape[1] != len(class_names):
            raise ValueError(
                f"model must return shape (batch, {len(class_names)}), got {tuple(logits.shape)}"
            )
        probabilities = logits.softmax(dim=1)
        for row in probabilities:
            index = int(row.argmax())
            results.append({
                "index": index,
                "label": str(class_names[index]),
                "confidence": float(row[index]),
                "scores": {str(name): float(row[i]) for i, name in enumerate(class_names)},
            })
    return results

