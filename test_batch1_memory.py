"""Run one full-resolution Copernicus training step to check GPU memory."""

import argparse
import time

import torch
import yaml

from libs.copernicus_h5_dataset import CopernicusH5Dataset
from libs.model import Model
from libs.utils import MaskedLpLoss, dict2namespace, normalize_uv_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="/data/copernicus_uv_data/processed_data/uovo_mid_1997-01-01_to_1997-12-31.h5")
    parser.add_argument("--config", default="configs/IFactFormer.yml")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = dict2namespace(yaml.safe_load(config_file))
    dataset = CopernicusH5Dataset(args.h5, input_days=config.model.in_time_window)
    try:
        x, y, target_valid, positions = dataset[args.index]
    finally:
        dataset.close()

    device = torch.device(args.device)
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    target_valid = target_valid.unsqueeze(0).to(device)
    positions = {key: value.unsqueeze(0).to(device) for key, value in positions.items()}
    model = Model(config.model).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    loss_fn = MaskedLpLoss(reduction=False)
    uv_mean = torch.zeros(2, device=device)
    uv_std = torch.ones(2, device=device)
    x = normalize_uv_input(x, uv_mean, uv_std)

    torch.cuda.reset_peak_memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    print(f"device={torch.cuda.get_device_name(device)} free_GiB={free / 2**30:.2f} total_GiB={total / 2**30:.2f}", flush=True)
    print(f"input={tuple(x.shape)} target={tuple(y.shape)} valid_pixels={target_valid.sum().item()}", flush=True)
    stage = "forward"
    start = time.perf_counter()
    try:
        prediction = model(x, positions)
        stage = "loss"
        loss = loss_fn(prediction, y, target_valid).mean()
        stage = "backward"
        loss.backward()
        stage = "optimizer"
        optimizer.step()
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as error:
        print(f"CUDA_OOM stage={stage}: {str(error).splitlines()[0]}", flush=True)
        raise
    finally:
        print(f"peak_allocated_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f} "
              f"peak_reserved_GiB={torch.cuda.max_memory_reserved(device) / 2**30:.2f}", flush=True)

    print(f"batch1_step_ok seconds={time.perf_counter() - start:.2f} loss={loss.item():.6f}", flush=True)


if __name__ == "__main__":
    main()
