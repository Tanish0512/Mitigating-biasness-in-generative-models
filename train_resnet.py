"""Train a ResNet-18 classifier on the repository's ImageFolder dataset.

The split is a reproducible, class-stratified random split. Eighty percent of
each class is used for training, ten percent for test, and ten percent for
validation. The exact assignment is saved to ``split.csv``.

Example:
    python train_resnet.py --data-dir images --output-dir resnet_runs --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import ImageFile
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(
    dataset: datasets.ImageFolder, seed: int
) -> tuple[list[int], list[int], list[int], dict[int, str]]:
    """Return a reproducible, class-stratified random 8:1:1 split."""
    by_class: dict[int, list[int]] = {class_index: [] for class_index in range(len(dataset.classes))}
    for index, (_, class_index) in enumerate(dataset.samples):
        by_class[class_index].append(index)

    train, validation, test = [], [], []
    assignments: dict[int, str] = {}
    for class_index, indices in by_class.items():
        shuffled = list(indices)
        random.Random(seed + class_index).shuffle(shuffled)
        test_count = len(shuffled) // 10
        validation_count = len(shuffled) // 10
        test_indices = shuffled[:test_count]
        validation_indices = shuffled[test_count:test_count + validation_count]
        train_indices = shuffled[test_count + validation_count:]
        test.extend(test_indices)
        validation.extend(validation_indices)
        train.extend(train_indices)
        for index in train_indices:
            assignments[index] = "train"
        for index in validation_indices:
            assignments[index] = "validation"
        for index in test_indices:
            assignments[index] = "test"
    return train, validation, test, assignments


def read_manifest_metadata(data_dir: Path, class_name: str) -> dict[str, dict[str, str]]:
    """Read optional source metadata from a class manifest."""
    manifest_path = data_dir / class_name / "manifest.csv"
    if not manifest_path.is_file():
        return {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return {row["filename"]: row for row in csv.DictReader(handle)}


def write_split_csv(
    path: Path,
    dataset: datasets.ImageFolder,
    assignments: dict[int, str],
) -> None:
    """Save the exact split and available source metadata."""
    manifests = {
        class_name: read_manifest_metadata(Path(dataset.root), class_name)
        for class_name in dataset.classes
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["path", "class", "source", "original_path", "sha256", "split"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (image_path, class_index) in enumerate(dataset.samples):
            class_name = dataset.classes[class_index]
            metadata = manifests[class_name].get(Path(image_path).name, {})
            writer.writerow({
                "path": str(Path(class_name) / Path(image_path).name),
                "class": class_name,
                "source": metadata.get("source", ""),
                "original_path": metadata.get("original_path", ""),
                "sha256": metadata.get("sha256", ""),
                "split": assignments[index],
            })


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    """Build ResNet-18 with a stem appropriate for 64x64 images."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if pretrained:
        # Preserve the useful central portion of the pretrained 7x7 filters
        # when changing the stem to a 3x3 filter.
        old_weight = model.conv1.weight.detach().clone()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        with torch.no_grad():
            model.conv1.weight.copy_(old_weight[:, :, 2:5, 2:5])
    else:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_loaders(
    dataset: datasets.ImageFolder,
    indices: tuple[list[int], list[int], list[int]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Each split gets its own ImageFolder instance so training augmentation is
    # never applied to validation or test images.
    train_dataset = Subset(datasets.ImageFolder(dataset.root, transform=train_transform), indices[0])
    validation_dataset = Subset(datasets.ImageFolder(dataset.root, transform=evaluation_transform), indices[1])
    test_dataset = Subset(datasets.ImageFolder(dataset.root, transform=evaluation_transform), indices[2])
    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_args),
        DataLoader(validation_dataset, shuffle=False, **loader_args),
        DataLoader(test_dataset, shuffle=False, **loader_args),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in tqdm(loader, desc="train" if training else "evaluate", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("resnet_runs"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Checkpoint to resume from, such as resnet_runs/checkpoint_last.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("image-size, batch-size, and epochs must be greater than zero")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")

    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    dataset = datasets.ImageFolder(args.data_dir)
    if not dataset.samples:
        raise RuntimeError(f"No images found in {args.data_dir}")

    train_indices, validation_indices, test_indices, assignments = split_indices(dataset, args.seed)
    split_csv = args.output_dir / "split.csv"
    write_split_csv(split_csv, dataset, assignments)
    loaders = make_loaders(
        dataset,
        (train_indices, validation_indices, test_indices),
        args.image_size,
        args.batch_size,
        args.num_workers,
        device,
    )
    # A resume checkpoint already contains all model weights, so it should not
    # trigger a fresh ImageNet-weight download on an offline server.
    model = build_model(len(dataset.classes), args.pretrained and args.resume is None).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classes.json").write_text(json.dumps(dataset.classes, indent=2) + "\n", encoding="utf-8")

    start_epoch = 1
    best_validation_accuracy = -1.0
    history: list[dict[str, float]] = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(f"{args.resume} is not a valid ResNet checkpoint")
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("classes") != dataset.classes:
            raise ValueError(
                "Checkpoint classes do not match the current dataset: "
                f"checkpoint={checkpoint.get('classes')}, dataset={dataset.classes}"
            )
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_validation_accuracy = float(checkpoint.get("best_validation_accuracy", -1.0))
        history = list(checkpoint.get("history", []))
        print(f"Resuming from {args.resume} at epoch {start_epoch}")

    print(
        f"Classes={dataset.classes}, train={len(train_indices):,}, "
        f"validation={len(validation_indices):,}, test={len(test_indices):,}, device={device}"
    )
    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}, but --epochs is {args.epochs}. "
            "Set --epochs to a larger final epoch count."
        )
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, loaders[0], criterion, device, optimizer, scaler)
        validation_loss, validation_accuracy = run_epoch(model, loaders[1], criterion, device)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}"
        )
        checkpoint = {
            "model": model.state_dict(),
            "classes": dataset.classes,
            "epoch": epoch,
            "args": vars(args),
            "split": "seeded random stratified split: 80% train, 10% test, 10% validation per class",
            "split_csv": str(split_csv),
            "validation_accuracy": validation_accuracy,
            "best_validation_accuracy": max(best_validation_accuracy, validation_accuracy),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
        }
        torch.save(checkpoint, args.output_dir / "checkpoint_last.pt")
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(checkpoint, args.output_dir / "checkpoint_best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    best = torch.load(args.output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_loss, test_accuracy = run_epoch(model, loaders[2], criterion, device)
    print(f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}")
    best["test_loss"] = test_loss
    best["test_accuracy"] = test_accuracy
    torch.save(best, args.output_dir / "checkpoint_best.pt")


if __name__ == "__main__":
    main()
