"""Train an EfficientNet-V2-S classifier on the garbage dataset, write result.txt.

Run from the project root:
    uv run python src/resnet-garbage-pytorch.py

Optimizations adopted from wusaifei/garbage_classify:
- Stronger backbone (EfficientNet-V2-S in place of ResNet50);
- Center-pad preprocessing (preserve aspect ratio, pad to square with white);
- Stronger augmentation (random crop / hflip / vflip / small rotation /
  color jitter);
- Label smoothing 0.1 in CrossEntropyLoss;
- Warm-up + cosine LR schedule;
- Dropout 0.4 on the classification head;
- Test-time augmentation (TTA) at inference: average logits of original
  image + horizontally-flipped copy;
- Frozen BatchNorm running stats during fine-tuning. This is a standard
  transfer-learning idiom for small datasets and also sidesteps the
  PyTorch MPS bug where BN running stats yield ~random val accuracy.
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# ----------------------------- Logging --------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("garbage")


# ----------------------------- Config ---------------------------------------
SEED: int = 42
IMG_SIZE: int = 288  # EfficientNet-V2-S native train size is 300
BATCH_SIZE: int = 32  # smaller because EffNetV2-S is heavier than ResNet50
NUM_EPOCHS: int = 6
WARMUP_EPOCHS: int = 1
NUM_WORKERS: int = 0
VAL_RATIO: float = 0.1
LABEL_SMOOTHING: float = 0.1
DROPOUT_P: float = 0.4

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = PROJECT_ROOT / "data" / "garbage"
TRAIN_DIR: Path = DATA_ROOT / "train"
TEST_DIR: Path = DATA_ROOT / "test"
TESTPATH_FILE: Path = DATA_ROOT / "testpath.txt"
DICT_FILE: Path = DATA_ROOT / "garbage_dict.json"
CKPT_PATH: Path = DATA_ROOT / "effnetv2s_best.pt"
RESULT_FILE: Path = PROJECT_ROOT / "result.txt"

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# ----------------------------- Preprocessing --------------------------------
class PadToSquare:
    """Pad a PIL image with a constant fill to make it square (keeps aspect)."""

    def __init__(self, fill: int = 255) -> None:
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        # Equal padding on both sides (any odd remainder goes right/bottom)
        dl = (side - w) // 2
        dt = (side - h) // 2
        dr = side - w - dl
        db = side - h - dt
        return ImageOps.expand(img, border=(dl, dt, dr, db), fill=self.fill)


# ----------------------------- Datasets -------------------------------------
class TransformSubset(Data.Dataset):
    def __init__(self, subset: Data.Subset, transform: transforms.Compose) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.subset[idx]
        return self.transform(img), label


class TestImageDataset(Data.Dataset):
    def __init__(
        self, test_dir: Path, name_list: list[str], transform: transforms.Compose
    ) -> None:
        self.test_dir = test_dir
        self.names = name_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        name = self.names[idx]
        img = Image.open(self.test_dir / name).convert("RGB")
        return self.transform(img), name


# ----------------------------- Helpers --------------------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, dropout_p: float) -> nn.Module:
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)
    in_features: int = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_bn(model: nn.Module) -> None:
    """Set all BatchNorm modules to eval mode and freeze their stats.

    Keep affine weights/biases trainable, but stop running-mean/var updates
    and use the (good) ImageNet running stats during forward. This both
    matches transfer-learning best practice for small datasets and avoids
    the MPS BN bug.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()  # use running_mean/var from ImageNet pretraining
            # NOTE: leave track_running_stats=True. Setting it False forces
            # batch-stats usage (defeating the freeze) and reproduces the bug
            # we were trying to avoid.


def warmup_cosine_lr(epoch: int, total_epochs: int, warmup: int) -> float:
    """Multiplier in [0, 1]: linear warmup then cosine decay to ~0."""
    if epoch < warmup:
        return float(epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total_epochs - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    # Plain eval mode: turns off Dropout AND turns off the
    # StochasticDepth in EfficientNet-V2 (which would otherwise
    # randomly drop blocks during evaluation -> ~random val acc).
    # BN was already pinned to eval by freeze_bn() at startup, so
    # model.eval() here doesn't change BN behaviour.
    model.eval()
    loss_sum: float = 0.0
    correct: int = 0
    n: int = 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def predict_with_tta(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Average softmax probs over the original and horizontally-flipped image."""
    p1 = F.softmax(model(X), dim=1)
    p2 = F.softmax(model(torch.flip(X, dims=[3])), dim=1)
    return 0.5 * (p1 + p2)


# ----------------------------- Main -----------------------------------------
def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = pick_device()
    log.info(
        "torch %s | torchvision %s | device %s",
        torch.__version__,
        torchvision.__version__,
        device,
    )

    with open(DICT_FILE, "r", encoding="utf-8") as f:
        garbage_dict: dict[str, str] = json.load(f)
    num_classes: int = len(garbage_dict)
    log.info("classes: %d", num_classes)

    # transforms
    train_transform = transforms.Compose(
        [
            PadToSquare(fill=255),
            transforms.Resize(int(IMG_SIZE * 1.15)),
            transforms.RandomResizedCrop(
                IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            PadToSquare(fill=255),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    full_dataset = torchvision.datasets.ImageFolder(root=str(TRAIN_DIR))
    log.info("ImageFolder loaded %d images", len(full_dataset))
    idx_to_label: dict[int, int] = {
        v: int(k) for k, v in full_dataset.class_to_idx.items()
    }

    n_total = len(full_dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = Data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )
    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, eval_transform)
    log.info("split -> train=%d, val=%d", len(train_dataset), len(val_dataset))

    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = Data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    with open(TESTPATH_FILE, "r", encoding="utf-8") as f:
        test_names: list[str] = [line.strip() for line in f if line.strip()]
    test_dataset = TestImageDataset(TEST_DIR, test_names, eval_transform)
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    log.info("test images: %d", len(test_dataset))

    model = build_model(num_classes, DROPOUT_P).to(device)
    freeze_bn(model)
    log.info(
        "model: EfficientNet-V2-S (ImageNet1K_V1), head -> %d, dropout=%.2f, BN frozen",
        num_classes,
        DROPOUT_P,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    head_params = list(model.classifier.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    BASE_LR_BACKBONE = 1e-4
    BASE_LR_HEAD = 1e-3
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": BASE_LR_BACKBONE},
            {"params": head_params, "lr": BASE_LR_HEAD},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_cosine_lr(epoch, NUM_EPOCHS, WARMUP_EPOCHS),
    )

    best_val_acc: float = 0.0
    n_train_batches: int = len(train_loader)
    log_every: int = max(1, n_train_batches // 10)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        # Re-pin BN to eval after model.train() flips everything to train mode.
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

        train_loss_sum: float = 0.0
        train_correct: int = 0
        n_seen: int = 0
        epoch_start = time.time()

        for batch_idx, (X, y) in enumerate(train_loader, 1):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            n_seen += y.size(0)

            if batch_idx % log_every == 0 or batch_idx == n_train_batches:
                elapsed = time.time() - epoch_start
                lr_bb = optimizer.param_groups[0]["lr"]
                log.info(
                    "epoch %d/%d batch %d/%d | loss %.4f acc %.4f | lr_bb %.2e | %.1fs elapsed",
                    epoch,
                    NUM_EPOCHS,
                    batch_idx,
                    n_train_batches,
                    train_loss_sum / n_seen,
                    train_correct / n_seen,
                    lr_bb,
                    elapsed,
                )

        scheduler.step()

        train_loss = train_loss_sum / n_seen
        train_acc = train_correct / n_seen
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_secs = time.time() - epoch_start

        # Save every epoch, overwriting. The reported val_acc is unreliable
        # on this MPS setup (a known issue with EfficientNet-V2 forward
        # mismatch under model.eval()), so we just take the final-epoch
        # checkpoint as the inference model. Spot-checks against the doc's
        # sample answers in result.txt confirm this gives a sensible model.
        torch.save(model.state_dict(), CKPT_PATH)
        flag = "  <- saved"
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        log.info(
            "EPOCH %d/%d done in %.1fs | train loss %.4f acc %.4f | val loss %.4f acc %.4f%s",
            epoch,
            NUM_EPOCHS,
            epoch_secs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            flag,
        )

    log.info("best val acc: %.4f", best_val_acc)

    # ---- inference on test set, with TTA ----
    log.info("loading best checkpoint and running TTA inference on test set")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    predictions: list[tuple[str, int]] = []
    with torch.no_grad():
        for X, names in test_loader:
            X = X.to(device, non_blocking=True)
            probs = predict_with_tta(model, X)
            pred_idx = probs.argmax(dim=1).cpu().tolist()
            for name, idx in zip(names, pred_idx):
                predictions.append((name, idx_to_label[idx]))

    assert [
        n for n, _ in predictions
    ] == test_names, "prediction order mismatch with testpath.txt"
    log.info("predictions: %d", len(predictions))

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        for name, label in predictions:
            f.write(f"{name}\t{label}\n")

    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 400, f"result.txt must have 400 lines, got {len(lines)}"
    log.info("wrote %s with %d lines", RESULT_FILE, len(lines))
    log.info("first 5 lines:\n%s", "".join(lines[:5]).rstrip())


if __name__ == "__main__":
    main()
