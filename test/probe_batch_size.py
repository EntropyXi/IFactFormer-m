"""Run one real-H5 optimizer step to measure single-GPU training memory.

Example:
    CUDA_VISIBLE_DEVICES=7 python test/probe_batch_size.py --batch-size 8 \
        --activation-checkpoint

Each invocation uses a fresh process, so peak allocator measurements do not
carry over from earlier batch-size trials. The same ocean-containing crop is
repeated within the batch; tensor shapes and backward memory match training.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from torch.utils.data._utils.collate import default_collate
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.copernicus_h5_dataset import CopernicusH5Dataset
from libs.model import Model
from libs.training_common import move_batch
from libs.utils import MaskedLpLoss, dict2namespace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=Path(
        "/data/copernicus_uv_data/processed_data/uovo_mid_1997-01-01_to_1997-12-31.h5"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/IFactFormer.yml")
    parser.add_argument("--patches-per-day", type=int, default=131)
    parser.add_argument("--window-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activation-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    return args


def select_ocean_window(dataset):
    for patch in (81, 0, 50, 100):
        if patch >= dataset.patches_per_day:
            continue
        for tile in range(dataset.tiles_per_patch):
            index = patch * dataset.tiles_per_patch + tile
            sample = dataset[index]
            x, _, valid, _ = sample
            if (valid & x[..., 2].bool().all(dim=0)).any():
                return index, sample
    raise ValueError("No valid ocean window found among probe candidates")


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    dataset = CopernicusH5Dataset(args.h5, patches_per_day=args.patches_per_day,
                                  input_days=config["model"]["in_time_window"],
                                  window_size=args.window_size)
    try:
        sample_index, sample = select_ocean_window(dataset)
        batch = default_collate([sample] * args.batch_size)
    finally:
        dataset.close()

    torch.manual_seed(1234)
    model = Model(dict2namespace(config["model"]),
                  activation_checkpoint=args.activation_checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    loss_fn = MaskedLpLoss(reduction=False)
    uv_mean = torch.zeros(2, device=device)
    uv_std = torch.ones(2, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = perf_counter()
    try:
        x, y, valid, positions = move_batch(batch, device, uv_mean, uv_std)
        prediction = model(x, positions) * uv_std + uv_mean
        losses = loss_fn(prediction, y, valid)
        loss = losses.sum() / valid.flatten(1).any(dim=1).sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = {"status": "ok", "batch_size": args.batch_size,
                  "sample_index": sample_index, "input_shape": list(x.shape),
                  "output_shape": list(prediction.shape),
                  "activation_checkpoint": args.activation_checkpoint,
                  "seconds": round(perf_counter() - started, 3),
                  "loss": float(loss.item())}
        if device.type == "cuda":
            result.update({
                "gpu": torch.cuda.get_device_name(device),
                "memory_total_gib": round(torch.cuda.get_device_properties(device).total_memory / 2**30, 3),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 3),
            })
        print(json.dumps(result), flush=True)
    except torch.cuda.OutOfMemoryError:
        result = {"status": "oom", "batch_size": args.batch_size,
                  "sample_index": sample_index,
                  "activation_checkpoint": args.activation_checkpoint,
                  "seconds": round(perf_counter() - started, 3)}
        if device.type == "cuda":
            result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            result["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
        print(json.dumps(result), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
