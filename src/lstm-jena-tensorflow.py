"""基于 TensorFlow/Keras 的 LSTM 模型对 Jena 气候数据进行温度预测。

完整的端到端实验流程：
  1. 加载 Jena Climate (2009-2016) 数据并完成降采样、标准化；
  2. 通过 ``timeseries_dataset_from_array`` 构造滑动窗口的训练 / 验证集；
  3. 定义 LSTM 网络（输入 -> LSTM(32) -> Dense(1)）并使用 Adam + MSE 训练；
  4. 绘制损失曲线与若干样本的「历史 / 真值 / 预测」对比图；
  5. 进行历史窗口长度（sequence_length）与隐藏单元数（units）的敏感度分析。

所有图像统一保存至 ``docs/lstm-jena-tensorflow_files/``，方便实验报告引用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# ----------------------------------------------------------------------------
# 0. 全局设置
# ----------------------------------------------------------------------------

PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "jena_climate_2009_2016.csv")
FIG_DIR: str = os.path.join(PROJECT_ROOT, "docs", "lstm-jena-tensorflow_files")
CKPT_DIR: str = os.path.join(PROJECT_ROOT, "data", "lstm_checkpoints")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# 14 个特征及对应的列名 / 颜色（用于原始可视化）
TITLES: List[str] = [
    "Pressure",
    "Temperature",
    "Temperature in Kelvin",
    "Temperature (dew point)",
    "Relative Humidity",
    "Saturation vapor pressure",
    "Vapor pressure",
    "Vapor pressure deficit",
    "Specific humidity",
    "Water vapor concentration",
    "Airtight",
    "Wind speed",
    "Maximum wind speed",
    "Wind direction in degrees",
]
FEATURE_KEYS: List[str] = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]
COLORS: List[str] = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
DATE_TIME_KEY: str = "Date Time"

# 经过相关性热图筛选后保留的 7 个特征索引
SELECTED_FEATURE_IDX: List[int] = [0, 1, 5, 7, 8, 10, 11]

# 基线超参数
SPLIT_FRACTION: float = 0.715
STEP: int = 6  # 降采样：每 6 个原始点（=1 小时）取 1 个
PAST: int = 720  # 历史窗口：720 小时 = 30 天
FUTURE: int = 72  # 预测目标：72 小时（3 天）后的温度
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 256
EPOCHS: int = 10

# 复现性
keras.utils.set_random_seed(42)


# ----------------------------------------------------------------------------
# 1. 数据加载与预处理
# ----------------------------------------------------------------------------


def load_dataframe() -> pd.DataFrame:
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"未找到数据文件 {DATA_PATH}。"
            "请先下载 jena_climate_2009_2016.csv 并放入 data/ 目录。"
        )
    return pd.read_csv(DATA_PATH)


def normalize(data: np.ndarray, train_split: int) -> np.ndarray:
    """按训练集统计量做 Z-score 标准化。"""
    mu = data[:train_split].mean(axis=0)
    sigma = data[:train_split].std(axis=0)
    return (data - mu) / sigma


def fig_path(name: str) -> str:
    return os.path.join(FIG_DIR, name)


# ----------------------------------------------------------------------------
# 2. 原始数据可视化（14 个特征 + 相关性热图）
# ----------------------------------------------------------------------------


def show_raw_visualization(df: pd.DataFrame) -> None:
    time_data = df[DATE_TIME_KEY]
    fig, axes = plt.subplots(
        nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i, key in enumerate(FEATURE_KEYS):
        c = COLORS[i % len(COLORS)]
        t_data = df[key].copy()
        t_data.index = time_data
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title=f"{TITLES[i]} - {key}",
            rot=25,
        )
        ax.legend([TITLES[i]])
    fig.tight_layout()
    fig.savefig(
        fig_path("lstm-jena-tensorflow_raw.png"),
        dpi=120,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig)


def show_heatmap(df: pd.DataFrame) -> None:
    data_numeric = df.drop(columns=[DATE_TIME_KEY])
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.matshow(data_numeric.corr())
    ax.set_xticks(range(data_numeric.shape[1]))
    ax.set_xticklabels(data_numeric.columns, fontsize=10, rotation=90)
    ax.set_yticks(range(data_numeric.shape[1]))
    ax.set_yticklabels(data_numeric.columns, fontsize=10)
    ax.xaxis.tick_bottom()
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=10)
    ax.set_title("Feature Correlation Heatmap", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(
        fig_path("lstm-jena-tensorflow_heatmap.png"),
        dpi=120,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. 滑动窗口数据集构造
# ----------------------------------------------------------------------------


@dataclass
class WindowedData:
    dataset_train: tf.data.Dataset
    dataset_val: tf.data.Dataset
    sequence_length: int
    num_features: int
    val_inputs_targets: Tuple[np.ndarray, np.ndarray]


def build_windowed_datasets(
    df: pd.DataFrame,
    past: int = PAST,
    future: int = FUTURE,
    step: int = STEP,
    batch_size: int = BATCH_SIZE,
) -> WindowedData:
    """按教程描述构造训练 / 验证窗口数据集。"""
    train_split = int(SPLIT_FRACTION * df.shape[0])

    selected = [FEATURE_KEYS[i] for i in SELECTED_FEATURE_IDX]
    features = df[selected].copy()
    features.index = df[DATE_TIME_KEY]
    feature_values = normalize(features.values, train_split)
    features = pd.DataFrame(feature_values)

    train_data = features.loc[: train_split - 1]
    val_data = features.loc[train_split:]

    # 训练集
    start = past + future
    end = start + train_split
    x_train = train_data[[i for i in range(7)]].values
    y_train = features.iloc[start:end][[1]]  # 目标列 = 温度（在 selected 中索引 1）
    sequence_length = int(past / step)

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    # 验证集
    x_end = len(val_data) - past - future
    label_start = train_split + past + future
    x_val = val_data.iloc[:x_end][[i for i in range(7)]].values
    y_val = features.iloc[label_start:][[1]]

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    return WindowedData(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        sequence_length=sequence_length,
        num_features=x_train.shape[-1],
        val_inputs_targets=(x_val, y_val.values),
    )


# ----------------------------------------------------------------------------
# 4. 模型定义、编译与训练
# ----------------------------------------------------------------------------


def build_lstm_model(
    sequence_length: int, num_features: int, units: int = 32
) -> keras.Model:
    inputs = keras.layers.Input(shape=(sequence_length, num_features))
    lstm_out = keras.layers.LSTM(units)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def fit_model(
    model: keras.Model,
    data: WindowedData,
    epochs: int,
    tag: str,
) -> keras.callbacks.History:
    ckpt_path = os.path.join(CKPT_DIR, f"{tag}.weights.h5")
    es_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=5, restore_best_weights=True
    )
    modelckpt_callback = keras.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filepath=ckpt_path,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )
    return model.fit(
        data.dataset_train,
        epochs=epochs,
        validation_data=data.dataset_val,
        callbacks=[es_callback, modelckpt_callback],
    )


# ----------------------------------------------------------------------------
# 5. 可视化：训练曲线与样本预测对比
# ----------------------------------------------------------------------------


def plot_loss_curve(
    history: keras.callbacks.History,
    save_name: str,
    title: str = "Training and Validation Loss",
) -> None:
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, "o-", label="Training Loss")
    ax.plot(epochs, val_loss, "s-", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path(save_name), dpi=120)
    plt.close(fig)


def plot_predictions(
    model: keras.Model,
    data: WindowedData,
    save_name: str,
    num_samples: int = 5,
) -> None:
    """从验证集中取若干个窗口，绘制 history / true future / model prediction。"""
    fig, axes = plt.subplots(num_samples, 1, figsize=(11, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    sample_count = 0
    for x, y in data.dataset_val.take(1):
        pred = model.predict(x, verbose=0)
        for i in range(min(num_samples, x.shape[0])):
            ax = axes[sample_count]
            history_seq = x[i, :, 1].numpy()  # 第 1 维 = 温度
            t_history = np.arange(-len(history_seq), 0)
            ax.plot(t_history, history_seq, label="History", color="#4C72B0")
            ax.scatter(
                [FUTURE / STEP],
                [y[i].numpy()],
                label="True Future",
                marker="x",
                color="#C44E52",
                s=80,
                zorder=5,
            )
            ax.scatter(
                [FUTURE / STEP],
                [pred[i, 0]],
                label="Model Prediction",
                marker="o",
                color="#55A868",
                s=80,
                zorder=5,
            )
            ax.set_xlabel("Time step (hour, normalized window)")
            ax.set_ylabel("Temperature (z-score)")
            ax.set_title(f"Sample {sample_count + 1}")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper left")
            sample_count += 1
            if sample_count >= num_samples:
                break
        break
    fig.tight_layout()
    fig.savefig(fig_path(save_name), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 6. 超参数敏感度分析
# ----------------------------------------------------------------------------


def sensitivity_sequence_length(df: pd.DataFrame) -> Dict[int, keras.callbacks.History]:
    """对比 past = 24h（短窗口）与 past = 720h（30 天，基线）。"""
    candidates: List[int] = [24, 720]
    results: Dict[int, keras.callbacks.History] = {}
    for past in candidates:
        keras.utils.set_random_seed(42)
        data = build_windowed_datasets(df, past=past)
        model = build_lstm_model(data.sequence_length, data.num_features, units=32)
        print(f"\n[seq_len] past={past}h, sequence_length={data.sequence_length}")
        history = fit_model(model, data, epochs=EPOCHS, tag=f"seqlen_{past}")
        results[past] = history
    return results


def sensitivity_units(df: pd.DataFrame) -> Dict[int, keras.callbacks.History]:
    """对比 LSTM units = 32 (基线) 与 units = 64。"""
    candidates: List[int] = [32, 64]
    results: Dict[int, keras.callbacks.History] = {}
    data = build_windowed_datasets(df)
    for units in candidates:
        keras.utils.set_random_seed(42)
        model = build_lstm_model(data.sequence_length, data.num_features, units=units)
        print(f"\n[units] units={units}")
        history = fit_model(model, data, epochs=EPOCHS, tag=f"units_{units}")
        results[units] = history
    return results


def plot_sensitivity_loss(
    histories: Dict[int, keras.callbacks.History],
    label_prefix: str,
    save_name: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, hist in histories.items():
        ep = range(1, len(hist.history["loss"]) + 1)
        axes[0].plot(ep, hist.history["loss"], "o-", label=f"{label_prefix}={k}")
        axes[1].plot(ep, hist.history["val_loss"], "s-", label=f"{label_prefix}={k}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss (MSE)")
    axes[0].set_title(f"{title} — Training Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss (MSE)")
    axes[1].set_title(f"{title} — Validation Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_path(save_name), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 7. 主流程
# ----------------------------------------------------------------------------


def main() -> None:
    print(f"TensorFlow 版本: {tf.__version__}")
    df = load_dataframe()
    print(f"原始数据形状: {df.shape}")
    print(f"训练集 / 验证集分割点: {int(SPLIT_FRACTION * df.shape[0])}")

    # 7.1 原始数据可视化 ----------------------------------------------------
    print("\n=== 原始数据可视化 ===")
    show_raw_visualization(df)
    show_heatmap(df)

    selected_titles = ", ".join([TITLES[i] for i in SELECTED_FEATURE_IDX])
    print(f"已选用特征: {selected_titles}")

    # 7.2 基线模型训练 -----------------------------------------------------
    print("\n=== 基线 LSTM(units=32, past=720h) ===")
    data = build_windowed_datasets(df)

    # 打印一个 batch 的形状信息
    for batch in data.dataset_train.take(1):
        inputs_b, targets_b = batch
        print(f"Input shape: {tuple(inputs_b.numpy().shape)}")
        print(f"Target shape: {tuple(targets_b.numpy().shape)}")

    model = build_lstm_model(data.sequence_length, data.num_features, units=32)
    model.summary()

    history = fit_model(model, data, epochs=EPOCHS, tag="baseline")
    plot_loss_curve(
        history,
        save_name="lstm-jena-tensorflow_loss.png",
        title="Baseline LSTM — Training and Validation Loss",
    )
    plot_predictions(model, data, save_name="lstm-jena-tensorflow_predictions.png")

    best_val_loss = min(history.history["val_loss"])
    print(f"基线模型最佳验证 MSE: {best_val_loss:.4f}")

    # 7.3 历史窗口长度敏感度 -----------------------------------------------
    print("\n=== 敏感度分析：历史窗口 sequence_length ===")
    seq_hist = sensitivity_sequence_length(df)
    plot_sensitivity_loss(
        seq_hist,
        label_prefix="past",
        save_name="lstm-jena-tensorflow_seqlen.png",
        title="Sequence Length Sensitivity",
    )

    # 7.4 隐藏单元数敏感度 ------------------------------------------------
    print("\n=== 敏感度分析：LSTM 隐藏单元数 units ===")
    units_hist = sensitivity_units(df)
    plot_sensitivity_loss(
        units_hist,
        label_prefix="units",
        save_name="lstm-jena-tensorflow_units.png",
        title="LSTM Units Sensitivity",
    )

    # 7.5 汇总 -----------------------------------------------------------
    print("\n=== 实验汇总（最佳验证集 MSE）===")
    print(f"baseline            : {best_val_loss:.4f}")
    for k, h in seq_hist.items():
        print(f"past={k:>3}h         : {min(h.history['val_loss']):.4f}")
    for k, h in units_hist.items():
        print(f"units={k:>3}         : {min(h.history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
