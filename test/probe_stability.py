"""Stress-test real windowed H5 batches without starting a full training run.

Reads distinct training windows through the normal DataLoader, then performs
repeated forward/backward/AdamW steps. No model checkpoint is saved.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
for directory in (str(ROOT), str(TEST_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from libs.copernicus_h5_dataset import CopernicusH5Dataset
from libs.model import Model
from libs.training_common import make_loader, move_batch, split_counts
from libs.utils import MaskedLpLoss, dict2namespace
from probe_batch_size import select_ocean_window


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=Path(
        "/data/copernicus_uv_data/processed_data/uovo_mid_1997-01-01_to_1997-12-31.h5"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/IFactFormer.yml")
    parser.add_argument("--stats", type=Path, required=True,
                        help="Training-only U/V statistics saved by the 448x448 run")
    parser.add_argument("--patches-per-day", type=int, default=131)
    parser.add_argument("--window-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activation-checkpoint", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.log_every) < 1 or args.num_workers < 0:
        parser.error("batch-size, steps and log-every must be positive; num-workers must be nonnegative")
    return args


def emit(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This stability probe requires CUDA")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.batch_size != config["training"]["batch_size"]:
        raise ValueError("Probe batch size must match the saved training config")
    dataset = CopernicusH5Dataset(args.h5, patches_per_day=args.patches_per_day,
                                  input_days=config["model"]["in_time_window"],
                                  window_size=args.window_size)
    try:
        train_samples = split_counts(dataset)[0]
        first_index, _ = select_ocean_window(dataset)
        requested = args.steps * args.batch_size
        if first_index + requested > train_samples:
            raise ValueError("Requested steps exceed the training split")
        loader = make_loader(dataset, range(first_index, first_index + requested),
                             args.batch_size, args.num_workers, device, seed=1234)
        stats = torch.load(args.stats, map_location="cpu", weights_only=True)
        mean, std = stats["mean"].to(device), stats["std"].to(device)
        if mean.shape != (2,) or std.shape != (2,) or not (std > 0).all():
            raise ValueError("Invalid U/V statistics")
        if stats.get("input_days") != dataset.input_days:
            raise ValueError("Statistics input_days differs from dataset")
        # Full 4x4 tiling weights every original pixel exactly once. Thus the
        # existing training-only 448x448 statistics apply to the same day split.
        if stats.get("train_samples") * dataset.tiles_per_patch != train_samples:
            raise ValueError("Statistics training-day split differs from dataset")

        torch.manual_seed(1234)
        model = Model(dict2namespace(config["model"]),
                      activation_checkpoint=args.activation_checkpoint).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
        loss_fn = MaskedLpLoss(reduction=False)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = perf_counter()
        completed = updated = 0
        first_loss = last_loss = None
        emit(event="start", batch_size=args.batch_size, steps=args.steps,
             first_sample_index=first_index, distinct_samples=requested,
             window_size=args.window_size, n_layer=config["model"]["n_layer"],
             activation_checkpoint=args.activation_checkpoint)
        try:
            for completed, batch in enumerate(loader, start=1):
                step_started = perf_counter()
                x, y, valid, positions = move_batch(batch, device, mean, std)
                prediction = model(x, positions) * std + mean
                losses = loss_fn(prediction, y, valid)
                valid_samples = valid.flatten(1).any(dim=1).sum()
                loss = losses.sum() / valid_samples.clamp_min(1)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {completed}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if valid_samples.item():
                    optimizer.step()
                    updated += 1
                torch.cuda.synchronize(device)
                last_loss = float(loss.item())
                if first_loss is None:
                    first_loss = last_loss
                if completed == 1 or completed % args.log_every == 0 or completed == args.steps:
                    emit(event="step", step=completed, valid_samples=int(valid_samples.item()),
                         loss=round(last_loss, 6),
                         step_seconds=round(perf_counter() - step_started, 3),
                         allocated_gib=round(torch.cuda.memory_allocated(device) / 2**30, 3),
                         reserved_gib=round(torch.cuda.memory_reserved(device) / 2**30, 3),
                         peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                         peak_reserved_gib=round(torch.cuda.max_memory_reserved(device) / 2**30, 3))
        except torch.cuda.OutOfMemoryError:
            emit(event="summary", status="oom", completed_steps=completed,
                 peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                 peak_reserved_gib=round(torch.cuda.max_memory_reserved(device) / 2**30, 3))
            return 2
        if completed != args.steps:
            raise RuntimeError(f"Only completed {completed}/{args.steps} steps")
        emit(event="summary", status="ok", completed_steps=completed,
             optimizer_updates=updated, first_loss=first_loss, last_loss=last_loss,
             total_seconds=round(perf_counter() - started, 3),
             peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
             peak_reserved_gib=round(torch.cuda.max_memory_reserved(device) / 2**30, 3))
        return 0
    finally:
        dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
