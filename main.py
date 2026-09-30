"""Train the 2-D IFactFormer-m on Copernicus H5 data with resumable checkpoints.

Start a new run with ``python -u main.py``. Resume an interrupted run with
``python -u main.py --resume results/<run>/checkpoint_latest.pt``.
For four-GPU DDP, use ``torchrun --standalone --nproc-per-node=4 main.py --ddp``.
DDP checkpoints must be resumed with the same world size and ``--ddp``.
"""

import argparse
from pathlib import Path
import random
import shutil
import signal
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

from libs.copernicus_h5_dataset import CopernicusH5Dataset, compute_train_uv_stats
from libs.model import Model
from libs.training_common import (atomic_save, make_loader, make_logger,
                                  move_batch, new_run_directory, split_counts)
from libs.utils import MaskedLpLoss, dict2namespace


DEFAULT_H5 = "/data/copernicus_uv_data/processed_data/uovo_mid_1997-01-01_to_1997-12-31.h5"
CHECKPOINT_NAME = "checkpoint_latest.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/IFactFormer.yml")
    parser.add_argument("--h5", default=DEFAULT_H5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ddp", action="store_true", help="Use torchrun for multi-process DDP training")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--resume", help="Latest checkpoint path or its run directory")
    parser.add_argument("--run-dir", help="Directory for a new run; must not exist")
    parser.add_argument("--patches-per-day", type=int, default=131)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stop-after-steps", type=int, help="Stop this invocation after N optimizer steps; useful for a smoke test")
    args = parser.parse_args()
    if args.resume and args.run_dir:
        parser.error("--resume and --run-dir cannot be used together")
    if args.num_workers < 0 or args.checkpoint_every_steps < 1 or args.log_every_steps < 1:
        parser.error("worker count must be nonnegative and step intervals must be positive")
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        parser.error("--stop-after-steps must be positive")
    return args


def evaluate(model, loader, loss_fn, device, uv_mean, uv_std, stop_requested):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            x, y, valid, positions = move_batch(batch, device, uv_mean, uv_std)
            prediction = model(x, positions) * uv_std + uv_mean
            losses = loss_fn(prediction, y, valid)
            loss_sum += losses.sum().item()
            sample_count += losses.numel()
            if stop_requested["value"]:
                return None
    return loss_sum / sample_count


def save_latest(path, model, optimizer, scheduler, config, args, h5_path, splits,
                uv_mean, uv_std, epoch, sample_offset, global_step, best_val, test_loss):
    checkpoint = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_config": config["model"],
        "training_config": config["training"],
        "h5_path": str(h5_path),
        "splits": splits,
        "patches_per_day": args.patches_per_day,
        "seed": args.seed,
        "uv_mean": uv_mean.detach().cpu(),
        "uv_std": uv_std.detach().cpu(),
        "epoch": epoch,
        "sample_offset": sample_offset,
        "global_step": global_step,
        "best_val": best_val,
        "test_loss": test_loss,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(args.device) if torch.device(args.device).type == "cuda" else None,
    }
    atomic_save(checkpoint, path)


def verify_resume(checkpoint, config, args, h5_path, splits):
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    if checkpoint["model_config"] != config["model"]:
        raise ValueError("Model config differs from the checkpoint")
    for name in ("batch_size", "lr", "scheduler_step", "scheduler_gamma"):
        if checkpoint["training_config"][name] != config["training"][name]:
            raise ValueError(f"Training setting {name} differs from the checkpoint")
    if checkpoint["h5_path"] != str(h5_path) or tuple(checkpoint["splits"]) != splits:
        raise ValueError("Dataset path or train/validation/test split differs from the checkpoint")
    if checkpoint["patches_per_day"] != args.patches_per_day or checkpoint["seed"] != args.seed:
        raise ValueError("Patch count or shuffle seed differs from the checkpoint")
    if checkpoint["epoch"] > config["training"]["epochs"]:
        raise ValueError("Configured epochs are fewer than completed epochs")
    if not 0 <= checkpoint["sample_offset"] <= splits[0]:
        raise ValueError("Invalid training sample offset in checkpoint")


def run(args):
    if args.ddp:
        from DDP.train import run_ddp
        return run_ddp(args)
    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if config["training"]["epochs"] < 1 or config["training"]["batch_size"] < 1:
        raise ValueError("Epoch count and batch size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    h5_path = Path(args.h5).resolve()
    dataset = CopernicusH5Dataset(h5_path, patches_per_day=args.patches_per_day,
                                  input_days=config["model"]["in_time_window"])
    splits = split_counts(dataset)
    ntrain, nval, ntest = splits

    if args.resume:
        latest_path = Path(args.resume).resolve()
        if latest_path.is_dir():
            latest_path = latest_path / CHECKPOINT_NAME
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        verify_resume(checkpoint, config, args, h5_path, splits)
        run_dir = latest_path.parent
    else:
        checkpoint = None
        run_dir = new_run_directory(config, args.run_dir)
        shutil.copyfile(config_path, run_dir / config_path.name)
        shutil.copyfile(Path(__file__).resolve(), run_dir / "main.py")
        latest_path = run_dir / CHECKPOINT_NAME

    logger = make_logger(run_dir)
    logger.info("run_dir=%s train/val/test=%d/%d/%d batch_size=%d n_layer=%d epochs=%d",
                run_dir, ntrain, nval, ntest, config["training"]["batch_size"],
                config["model"]["n_layer"], config["training"]["epochs"])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if checkpoint is None:
        logger.info("Computing training-only ocean U/V statistics from H5")
        uv_mean, uv_std = compute_train_uv_stats(dataset, ntrain)
        atomic_save({"mean": uv_mean, "std": uv_std, "train_samples": ntrain,
                     "input_days": dataset.input_days}, run_dir / "uv_stats.pt")
        epoch = sample_offset = global_step = 0
        best_val = float("inf")
        test_loss = None
    else:
        uv_mean, uv_std = checkpoint["uv_mean"], checkpoint["uv_std"]
        epoch = checkpoint["epoch"]
        sample_offset = checkpoint["sample_offset"]
        global_step = checkpoint["global_step"]
        best_val = checkpoint["best_val"]
        test_loss = checkpoint["test_loss"]
        logger.info("Resuming epoch=%d sample_offset=%d global_step=%d", epoch, sample_offset, global_step)

    uv_mean, uv_std = uv_mean.to(device), uv_std.to(device)
    model = Model(dict2namespace(config["model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["training"]["scheduler_step"],
        gamma=config["training"]["scheduler_gamma"])
    loss_fn = MaskedLpLoss(reduction=False)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
    else:
        save_latest(latest_path, model, optimizer, scheduler, config, args, h5_path,
                    splits, uv_mean, uv_std, epoch, sample_offset, global_step, best_val, test_loss)

    stop_requested = {"value": False}

    def request_stop(signum, _frame):
        stop_requested["value"] = True
        stop_requested["signal"] = signum

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    steps_this_run = 0
    batch_size = config["training"]["batch_size"]
    best_path = run_dir / "checkpoint_best.pth"
    try:
        while epoch < config["training"]["epochs"]:
            model.train()
            started = perf_counter()
            order = torch.randperm(ntrain, generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
            train_loader = make_loader(dataset, order[sample_offset:], batch_size,
                                       args.num_workers, device, args.seed + epoch)
            for batch in train_loader:
                x, y, valid, positions = move_batch(batch, device, uv_mean, uv_std)
                prediction = model(x, positions) * uv_std + uv_mean
                loss = loss_fn(prediction, y, valid).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, step {global_step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                sample_offset += x.shape[0]
                global_step += 1
                steps_this_run += 1
                if global_step % args.log_every_steps == 0:
                    logger.info("epoch=%d samples=%d/%d step=%d loss=%.6f",
                                epoch, sample_offset, ntrain, global_step, loss.item())
                should_stop = stop_requested["value"] or (
                    args.stop_after_steps is not None and steps_this_run >= args.stop_after_steps)
                if global_step % args.checkpoint_every_steps == 0 or should_stop:
                    save_latest(latest_path, model, optimizer, scheduler, config, args, h5_path,
                                splits, uv_mean, uv_std, epoch, sample_offset, global_step, best_val, test_loss)
                    logger.info("checkpoint saved at epoch=%d samples=%d/%d", epoch, sample_offset, ntrain)
                if should_stop:
                    logger.info("Stopped safely; resume with --resume %s", latest_path)
                    return 0

            # A completed training epoch is saved before validation. If validation
            # is interrupted, resume will repeat validation without optimizer steps.
            save_latest(latest_path, model, optimizer, scheduler, config, args, h5_path,
                        splits, uv_mean, uv_std, epoch, ntrain, global_step, best_val, test_loss)
            val_loader = make_loader(dataset, range(ntrain, ntrain + nval), batch_size,
                                     args.num_workers, device, args.seed + 1000000 + epoch)
            val_loss = evaluate(model, val_loader, loss_fn, device, uv_mean, uv_std, stop_requested)
            if val_loss is None:
                logger.info("Validation interrupted; resume with --resume %s", latest_path)
                return 0
            if val_loss < best_val:
                best_val = val_loss
                atomic_save({"model": model.state_dict(), "epoch": epoch,
                             "val_loss": val_loss}, best_path)
            scheduler.step()
            logger.info("epoch=%d complete seconds=%.1f val_L2=%.6f best_val_L2=%.6f",
                        epoch, perf_counter() - started, val_loss, best_val)
            epoch += 1
            sample_offset = 0
            save_latest(latest_path, model, optimizer, scheduler, config, args, h5_path,
                        splits, uv_mean, uv_std, epoch, sample_offset, global_step, best_val, test_loss)

        if test_loss is None:
            best = torch.load(best_path, map_location=device, weights_only=True)
            final_train_state = {name: value.detach().cpu().clone()
                                 for name, value in model.state_dict().items()}
            model.load_state_dict(best["model"])
            test_loader = make_loader(dataset, range(ntrain + nval, len(dataset)), batch_size,
                                      args.num_workers, device, args.seed + 2000000)
            test_loss = evaluate(model, test_loader, loss_fn, device, uv_mean, uv_std, stop_requested)
            if test_loss is None:
                logger.info("Test interrupted; resume with --resume %s", latest_path)
                return 0
            logger.info("test_L2=%.6f best_epoch=%d", test_loss, best["epoch"])
            model.load_state_dict(final_train_state)
            save_latest(latest_path, model, optimizer, scheduler, config, args, h5_path,
                        splits, uv_mean, uv_std, epoch, 0, global_step, best_val, test_loss)
        else:
            logger.info("Run already completed: test_L2=%.6f", test_loss)
        return 0
    finally:
        dataset.close()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
