# SPDX-License-Identifier: MIT
"""Data loading and DataLoader creation for 1D signals (NPZ format).

NPZ schema:
    Pretrain file: {"signals": (n, signal_size) float32}
    Probing file:  {"signals": (n, signal_size) float32,
                    "<label>":  (n,) float32 (regression) or int (classification),
                    ...}  # any number of additional label arrays

Signals are expected to be roughly normalized to [0, 1].
"""

import platform

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def _auto_loader_kwargs():
    """Auto-detect num_workers and pin_memory based on platform and GPU."""
    if platform.system() == "Windows":
        return {"num_workers": 0, "pin_memory": torch.cuda.is_available()}
    if torch.cuda.is_available():
        return {"num_workers": 2, "pin_memory": True}
    return {"num_workers": 0, "pin_memory": False}


def load_signals_npz(filename):
    """Load 'signals' plus any other arrays (treated as labels) from an NPZ file."""
    data = np.load(filename, allow_pickle=False)
    signals = data["signals"]
    labels = {k: data[k] for k in data if k not in ("signals", "config")}
    return signals, labels


def load_and_split_pretrain(filename, split=(0.8, 0.1, 0.1), seed=42):
    """Load a pretrain NPZ and split deterministically into train/val/test."""
    signals, _ = load_signals_npz(filename)
    n = len(signals)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    signals = signals[idx]

    n_train = int(n * split[0])
    n_val = int(n * (split[0] + split[1]))

    x_train = signals[:n_train]
    x_val = signals[n_train:n_val]
    x_test = signals[n_val:]

    print(f"Pretrain split: {len(x_train)}/{len(x_val)}/{len(x_test)}")
    return x_train, x_val, x_test


def load_and_split_probing(filename, factor_name, split=(0.8, 0.1, 0.1), seed=42):
    """Load a probing NPZ, extract one label array, split deterministically."""
    signals, labels = load_signals_npz(filename)
    y = labels[factor_name]
    n = len(signals)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    signals = signals[idx]
    y = y[idx]

    n_train = int(n * split[0])
    n_val = int(n * (split[0] + split[1]))

    x_train, y_train = signals[:n_train], y[:n_train]
    x_val, y_val = signals[n_train:n_val], y[n_train:n_val]
    x_test, y_test = signals[n_val:], y[n_val:]

    print(f"Probing [{factor_name}]: {len(x_train)}/{len(x_val)}/{len(x_test)}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_pretrain_dataloaders(x_train, x_val, x_test, batch_size=1024,
                                preload_device=None):
    """DataLoaders for MAE pretraining (signals only).

    If preload_device is set, tensors are moved there once to avoid per-batch
    CPU->GPU transfers.
    """
    x_train_tensor = torch.from_numpy(x_train).float()
    x_val_tensor = torch.from_numpy(x_val).float()
    x_test_tensor = torch.from_numpy(x_test).float()

    if preload_device is not None:
        x_train_tensor = x_train_tensor.to(preload_device)
        x_val_tensor = x_val_tensor.to(preload_device)
        x_test_tensor = x_test_tensor.to(preload_device)
        loader_kw = {"num_workers": 0, "pin_memory": False}
    else:
        loader_kw = _auto_loader_kwargs()

    train_dataset = TensorDataset(x_train_tensor)
    val_dataset = TensorDataset(x_val_tensor)
    test_dataset = TensorDataset(x_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kw,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kw,
    )

    return train_loader, val_loader, test_loader


def create_downstream_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test,
                               batch_size=512, preload_device=None,
                               task_type="regression"):
    """DataLoaders for downstream training. task_type 'classification' casts labels to long."""
    x_train_t = torch.from_numpy(x_train).float()
    x_val_t = torch.from_numpy(x_val).float()
    x_test_t = torch.from_numpy(x_test).float()

    if task_type == "classification":
        y_train_t = torch.from_numpy(y_train.astype(np.int64)).long()
        y_val_t = torch.from_numpy(y_val.astype(np.int64)).long()
        y_test_t = torch.from_numpy(y_test.astype(np.int64)).long()
    else:
        y_train_t = torch.from_numpy(y_train).float()
        y_val_t = torch.from_numpy(y_val).float()
        y_test_t = torch.from_numpy(y_test).float()

    if preload_device is not None:
        x_train_t = x_train_t.to(preload_device)
        x_val_t = x_val_t.to(preload_device)
        x_test_t = x_test_t.to(preload_device)
        y_train_t = y_train_t.to(preload_device)
        y_val_t = y_val_t.to(preload_device)
        y_test_t = y_test_t.to(preload_device)
        loader_kw = {"num_workers": 0, "pin_memory": False}
    else:
        loader_kw = _auto_loader_kwargs()

    train_dataset = TensorDataset(x_train_t, y_train_t)
    val_dataset = TensorDataset(x_val_t, y_val_t)
    test_dataset = TensorDataset(x_test_t, y_test_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kw,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kw,
    )

    return train_loader, val_loader, test_loader
