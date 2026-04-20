"""基于 PyTorch 的卷积神经网络（CNN）识别 MNIST 手写数字。

该脚本为完整的端到端实验流程：
  1. 加载并预处理 MNIST 数据集；
  2. 定义 SimpleCNN 网络结构（两卷积层 + 两全连接层）；
  3. 使用交叉熵损失与 Adam 优化器训练 10 个 epoch；
  4. 在测试集上评估并可视化预测结果；
  5. 对批大小、学习率、卷积核数量三个关键超参数进行敏感度分析。

所有图像均保存至 ``docs/cnn-mnist-pytorch_files/``，以便实验报告直接引用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms


# ----------------------------------------------------------------------------
# 0. 全局设置
# ----------------------------------------------------------------------------

PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_ROOT: str = os.path.join(PROJECT_ROOT, "data")
FIG_DIR: str = os.path.join(PROJECT_ROOT, "docs", "cnn-mnist-pytorch_files")
os.makedirs(FIG_DIR, exist_ok=True)

MNIST_MEAN: Tuple[float, ...] = (0.1307,)
MNIST_STD: Tuple[float, ...] = (0.3081,)

# 基线超参数（对应实验教程中的推荐值）
BASELINE_BATCH_SIZE: int = 32
BASELINE_LR: float = 1e-3
BASELINE_CONV1_OUT: int = 32
BASELINE_CONV2_OUT: int = 64
NUM_EPOCHS: int = 10
SENSITIVITY_EPOCHS: int = 5

# 设置随机种子，保证不同运行之间的可复现性
torch.manual_seed(42)
np.random.seed(42)


def select_device() -> torch.device:
    """按 ``cuda -> mps -> cpu`` 的优先级选择训练设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# 1. 数据加载
# ----------------------------------------------------------------------------


def build_transform() -> transforms.Compose:
    """将 PIL 图像转为 Tensor 并按 MNIST 统计量标准化。"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def load_datasets() -> Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
    """加载 MNIST 训练集与测试集。原始 idx 文件若不存在将尝试下载。"""
    transform = build_transform()
    raw_dir: str = os.path.join(DATA_ROOT, "MNIST", "raw")
    need_download: bool = not os.path.isdir(raw_dir)

    train_set = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=True, download=need_download, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=DATA_ROOT, train=False, download=need_download, transform=transform
    )
    return train_set, test_set


def build_dataloaders(
    batch_size: int = BASELINE_BATCH_SIZE,
) -> Tuple[Data.DataLoader, Data.DataLoader]:
    train_set, test_set = load_datasets()
    train_loader = Data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = Data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


# ----------------------------------------------------------------------------
# 2. 模型定义
# ----------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """用于 MNIST 的简易 CNN：两卷积层 + 一次最大池化 + 两全连接层。

    输出特征图尺寸遵循 ``H_out = floor((H_in + 2P - K) / S) + 1``。
    对于默认设置（P=0, K=3, S=1, pool=2）：

        28 --conv1--> 26 --conv2--> 24 --pool--> 12
    """

    def __init__(
        self,
        conv1_out: int = BASELINE_CONV1_OUT,
        conv2_out: int = BASELINE_CONV2_OUT,
        fc_hidden: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=3)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten_dim: int = conv2_out * 12 * 12
        self.fc1 = nn.Linear(self.flatten_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征提取：两层卷积 + 一次最大池化
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # 展平后进入全连接分类头
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        # 直接输出 logits；CrossEntropyLoss 内部会做 LogSoftmax
        return self.fc2(x)


# ----------------------------------------------------------------------------
# 3. 训练与评估
# ----------------------------------------------------------------------------


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float


@dataclass
class TrainHistory:
    records: List[EpochRecord] = field(default_factory=list)

    @property
    def epochs(self) -> List[int]:
        return [r.epoch for r in self.records]

    @property
    def train_losses(self) -> List[float]:
        return [r.train_loss for r in self.records]

    @property
    def train_accs(self) -> List[float]:
        return [r.train_acc for r in self.records]

    @property
    def test_losses(self) -> List[float]:
        return [r.test_loss for r in self.records]

    @property
    def test_accs(self) -> List[float]:
        return [r.test_acc for r in self.records]


def train_one_epoch(
    model: nn.Module,
    device: torch.device,
    loader: Data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.train()
    loss_sum: float = 0.0
    correct: int = 0
    total: int = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits: torch.Tensor = model(X)
        batch_loss: torch.Tensor = criterion(logits, y)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss_sum += batch_loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: Data.DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    loss_sum: float = 0.0
    correct: int = 0
    total: int = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits: torch.Tensor = model(X)
        batch_loss: torch.Tensor = criterion(logits, y)
        loss_sum += batch_loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def fit(
    model: nn.Module,
    device: torch.device,
    train_loader: Data.DataLoader,
    test_loader: Data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    tag: str = "",
) -> TrainHistory:
    history = TrainHistory()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, device, train_loader, optimizer, criterion
        )
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        history.records.append(
            EpochRecord(epoch, train_loss, train_acc, test_loss, test_acc)
        )
        prefix: str = f"[{tag}] " if tag else ""
        print(
            f"{prefix}epoch {epoch:2d}, "
            f"train loss {train_loss:.4f}, train acc {train_acc:.4f}, "
            f"test loss {test_loss:.4f}, test acc {test_acc:.4f}"
        )
    return history


# ----------------------------------------------------------------------------
# 4. 可视化
# ----------------------------------------------------------------------------


def fig_path(name: str) -> str:
    return os.path.join(FIG_DIR, name)


def plot_sample_images(loader: Data.DataLoader, save_name: str, num: int = 8) -> None:
    X, y = next(iter(loader))
    fig, axes = plt.subplots(1, num, figsize=(num * 1.5, 2))
    for i, ax in enumerate(axes):
        img = X[i].squeeze().numpy() * MNIST_STD[0] + MNIST_MEAN[0]
        ax.imshow(np.clip(img, 0.0, 1.0), cmap="gray")
        ax.set_title(f"label: {y[i].item()}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_path(save_name), dpi=120)
    plt.close(fig)


def plot_learning_curves(history: TrainHistory, save_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.epochs, history.train_losses, marker="o", label="train")
    axes[0].plot(history.epochs, history.test_losses, marker="s", label="test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Loss vs. Epoch")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].plot(history.epochs, history.train_accs, marker="o", label="train")
    axes[1].plot(history.epochs, history.test_accs, marker="s", label="test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs. Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(fig_path(save_name), dpi=120)
    plt.close(fig)


def plot_sensitivity_curves(
    histories: Dict[str, TrainHistory], title: str, save_name: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, h in histories.items():
        axes[0].plot(h.epochs, h.train_losses, marker="o", label=f"{label}")
        axes[1].plot(h.epochs, h.test_accs, marker="s", label=f"{label}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title(f"{title} — Train Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title(f"{title} — Test Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path(save_name), dpi=120)
    plt.close(fig)


def plot_best_acc_bar(
    histories: Dict[str, TrainHistory], title: str, save_name: str
) -> None:
    labels = list(histories.keys())
    best_accs = [max(h.test_accs) for h in histories.values()]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, best_accs, color="#4C72B0")
    y_lo = max(0.0, min(best_accs) - 0.01)
    ax.set_ylim(y_lo, 1.0)
    ax.set_ylabel("Best Test Accuracy")
    ax.set_title(title)
    for bar, acc in zip(bars, best_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_path(save_name), dpi=120)
    plt.close(fig)


def plot_sample_predictions(
    model: nn.Module,
    device: torch.device,
    loader: Data.DataLoader,
    save_name: str,
    num: int = 16,
) -> None:
    model.eval()
    X, y = next(iter(loader))
    with torch.no_grad():
        preds = model(X.to(device)).argmax(dim=1).cpu()
    rows = 2
    cols = num // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2))
    for i, ax in enumerate(axes.flat):
        img = X[i].squeeze().numpy() * MNIST_STD[0] + MNIST_MEAN[0]
        ax.imshow(np.clip(img, 0.0, 1.0), cmap="gray")
        color = "green" if preds[i].item() == y[i].item() else "red"
        ax.set_title(
            f"pred: {preds[i].item()}\nlabel: {y[i].item()}",
            color=color,
            fontsize=9,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_path(save_name), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. 超参数敏感度分析
# ----------------------------------------------------------------------------


def run_single_config(
    device: torch.device,
    batch_size: int,
    lr: float,
    conv1_out: int,
    conv2_out: int,
    num_epochs: int,
    tag: str,
) -> TrainHistory:
    """以给定的超参数组合重新初始化网络并训练 ``num_epochs`` 轮。"""
    torch.manual_seed(42)  # 每次重置以保证不同配置之间公平比较
    train_loader, test_loader = build_dataloaders(batch_size=batch_size)
    model = SimpleCNN(conv1_out=conv1_out, conv2_out=conv2_out).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return fit(
        model,
        device,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        num_epochs=num_epochs,
        tag=tag,
    )


def sensitivity_batch_size(device: torch.device) -> Dict[str, TrainHistory]:
    candidates: List[int] = [32, 128, 512]
    histories: Dict[str, TrainHistory] = {}
    for bs in candidates:
        histories[f"bs={bs}"] = run_single_config(
            device=device,
            batch_size=bs,
            lr=BASELINE_LR,
            conv1_out=BASELINE_CONV1_OUT,
            conv2_out=BASELINE_CONV2_OUT,
            num_epochs=SENSITIVITY_EPOCHS,
            tag=f"bs={bs}",
        )
    return histories


def sensitivity_learning_rate(device: torch.device) -> Dict[str, TrainHistory]:
    candidates: List[float] = [1e-4, 1e-3, 1e-2]
    histories: Dict[str, TrainHistory] = {}
    for lr in candidates:
        histories[f"lr={lr:g}"] = run_single_config(
            device=device,
            batch_size=BASELINE_BATCH_SIZE,
            lr=lr,
            conv1_out=BASELINE_CONV1_OUT,
            conv2_out=BASELINE_CONV2_OUT,
            num_epochs=SENSITIVITY_EPOCHS,
            tag=f"lr={lr:g}",
        )
    return histories


def sensitivity_conv_filters(device: torch.device) -> Dict[str, TrainHistory]:
    # (conv1_out, conv2_out) 对：通道数保持 2 倍关系
    candidates: List[Tuple[int, int]] = [(16, 32), (32, 64), (64, 128)]
    histories: Dict[str, TrainHistory] = {}
    for c1, c2 in candidates:
        tag = f"filters={c1}/{c2}"
        histories[tag] = run_single_config(
            device=device,
            batch_size=BASELINE_BATCH_SIZE,
            lr=BASELINE_LR,
            conv1_out=c1,
            conv2_out=c2,
            num_epochs=SENSITIVITY_EPOCHS,
            tag=tag,
        )
    return histories


# ----------------------------------------------------------------------------
# 6. 主流程
# ----------------------------------------------------------------------------


def main() -> None:
    device = select_device()
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"使用设备: {device}")

    # 6.1 基线训练 ------------------------------------------------------------
    train_loader, test_loader = build_dataloaders(batch_size=BASELINE_BATCH_SIZE)
    plot_sample_images(train_loader, save_name="cnn-mnist-pytorch_samples.png")

    model = SimpleCNN().to(device)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"可学习参数总数: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=BASELINE_LR)
    criterion = nn.CrossEntropyLoss()

    history = fit(
        model,
        device,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        num_epochs=NUM_EPOCHS,
        tag="baseline",
    )
    plot_learning_curves(history, save_name="cnn-mnist-pytorch_curves.png")
    plot_sample_predictions(
        model, device, test_loader, save_name="cnn-mnist-pytorch_predictions.png"
    )

    best_test_acc = max(history.test_accs)
    print(f"基线模型最佳测试准确率: {best_test_acc:.4f}")

    # 6.2 超参数敏感度分析 ----------------------------------------------------
    print("\n=== 超参数敏感度：批大小 batch_size ===")
    bs_hist = sensitivity_batch_size(device)
    plot_sensitivity_curves(
        bs_hist, title="Batch Size", save_name="cnn-mnist-pytorch_bs_curves.png"
    )
    plot_best_acc_bar(
        bs_hist,
        title="Best Test Acc vs. Batch Size",
        save_name="cnn-mnist-pytorch_bs_bar.png",
    )

    print("\n=== 超参数敏感度：学习率 learning_rate ===")
    lr_hist = sensitivity_learning_rate(device)
    plot_sensitivity_curves(
        lr_hist, title="Learning Rate", save_name="cnn-mnist-pytorch_lr_curves.png"
    )
    plot_best_acc_bar(
        lr_hist,
        title="Best Test Acc vs. Learning Rate",
        save_name="cnn-mnist-pytorch_lr_bar.png",
    )

    print("\n=== 超参数敏感度：卷积核数量 conv_filters ===")
    cf_hist = sensitivity_conv_filters(device)
    plot_sensitivity_curves(
        cf_hist, title="Conv Filters", save_name="cnn-mnist-pytorch_cf_curves.png"
    )
    plot_best_acc_bar(
        cf_hist,
        title="Best Test Acc vs. Conv Filters",
        save_name="cnn-mnist-pytorch_cf_bar.png",
    )

    # 6.3 汇总 ---------------------------------------------------------------
    summary: Dict[str, Dict[str, float]] = {
        "baseline": {"best_test_acc": best_test_acc},
        "batch_size": {k: max(v.test_accs) for k, v in bs_hist.items()},
        "learning_rate": {k: max(v.test_accs) for k, v in lr_hist.items()},
        "conv_filters": {k: max(v.test_accs) for k, v in cf_hist.items()},
    }
    print("\n=== 实验汇总 ===")
    for group, results in summary.items():
        print(f"[{group}] {results}")


if __name__ == "__main__":
    main()
