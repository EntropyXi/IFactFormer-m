"""Small training helpers shared by the single-process and DDP entry points."""

import logging
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

from libs.utils import normalize_uv_input


def split_counts(dataset):
    window_days = dataset.num_days - dataset.input_days
    train_days = round(0.8 * window_days)
    val_days = round(0.1 * window_days)
    test_days = window_days - train_days - val_days
    if min(train_days, val_days, test_days) < 1:
        raise ValueError("Each split must contain at least one target day")
    samples_per_day = dataset.samples_per_day
    return (train_days * samples_per_day,
            val_days * samples_per_day,
            test_days * samples_per_day)


def new_run_directory(config, explicit):
    if explicit:
        run_dir = Path(explicit).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    root = Path(config["log_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model = config["model"]
    prefix = f"dim{model['dim']}_depth{model['depth']}_iter{model['n_layer']}_T{model['in_time_window']}"
    for index in range(10000):
        run_dir = root / f"{prefix}_{index}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError("No unused results directory was found")


def make_logger(run_dir):
    logger = logging.getLogger("ifactformer_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "training.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def make_loader(dataset, indices, batch_size, num_workers, device, seed):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),
    )


def move_batch(batch, device, uv_mean, uv_std):
    x, y, valid, positions = batch
    x = normalize_uv_input(x.to(device, non_blocking=True), uv_mean, uv_std)
    y = y.to(device, non_blocking=True)
    valid = valid.to(device, non_blocking=True)
    positions = {name: value.to(device, non_blocking=True) for name, value in positions.items()}
    return x, y, valid, positions
