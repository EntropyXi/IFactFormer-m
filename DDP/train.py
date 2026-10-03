"""Four-GPU-capable DDP runner, invoked only by main.py --ddp."""

from pathlib import Path
import random
import shutil
import signal
from time import perf_counter

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import yaml

from DDP.checkpoint import save_latest, verify_resume
from DDP.data import evaluation_indices, samples_per_rank, train_indices
from DDP.runtime import DistributedRuntime
from libs.copernicus_h5_dataset import CopernicusH5Dataset, compute_train_uv_stats
from libs.model import Model
from libs.training_common import (atomic_save, make_loader, make_logger,
                                  move_batch, new_run_directory, split_counts)
from libs.utils import MaskedLpLoss, dict2namespace


CHECKPOINT_NAME = "checkpoint_latest.pt"


def evaluate_distributed(model, loader, loss_fn, runtime, uv_mean, uv_std, stop_requested):
    """Evaluate unpadded shards; only aggregate scalar sum and count."""
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x, y, valid, positions = move_batch(batch, runtime.device, uv_mean, uv_std)
            prediction = model(x, positions) * uv_std + uv_mean
            losses = loss_fn(prediction, y, valid)
            loss_sum += losses.sum().item()
            count += valid.flatten(1).any(dim=1).sum().item()
            if stop_requested["value"]:
                break
    stopped = runtime.any_true(stop_requested["value"])
    if stopped:
        return None
    return runtime.weighted_mean(loss_sum, count)


def run_ddp(args):
    runtime = DistributedRuntime(args.device)
    try:
        return _run_ddp(args, runtime)
    finally:
        runtime.close()


def _run_ddp(args, runtime):
    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if config["training"]["epochs"] < 1 or config["training"]["batch_size"] < 1:
        raise ValueError("Epoch count and per-rank batch size must be positive")
    h5_path = Path(args.h5).resolve()
    dataset = CopernicusH5Dataset(h5_path, patches_per_day=args.patches_per_day,
                                  input_days=config["model"]["in_time_window"],
                                  window_size=args.window_size,
                                  tile_selection=args.tile_selection)
    try:
        splits = split_counts(dataset)
        ntrain, nval, ntest = splits
        per_rank = samples_per_rank(ntrain, runtime.world_size)
        padded_count = per_rank * runtime.world_size

        if args.resume:
            latest_path = Path(args.resume).resolve()
            if latest_path.is_dir():
                latest_path = latest_path / CHECKPOINT_NAME
            checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
            verify_resume(checkpoint, config, args, h5_path, splits, runtime, per_rank)
            run_dir = latest_path.parent
        else:
            checkpoint = None
            run_dir = None
            if runtime.primary:
                run_dir = new_run_directory(config, args.run_dir)
                shutil.copyfile(config_path, run_dir / config_path.name)
                shutil.copyfile(Path(__file__).resolve().parent.parent / "main.py", run_dir / "main.py")
                source_dir = run_dir / "DDP_source"
                source_dir.mkdir()
                for source in Path(__file__).resolve().parent.glob("*.py"):
                    shutil.copyfile(source, source_dir / source.name)
                shutil.copyfile(Path(__file__).resolve().parent.parent / "libs" / "training_common.py",
                                source_dir / "training_common.py")
            run_dir = Path(runtime.broadcast_object(str(run_dir) if runtime.primary else None))
            latest_path = run_dir / CHECKPOINT_NAME

        logger = make_logger(run_dir) if runtime.primary else None
        if runtime.primary:
            logger.info("DDP run_dir=%s world_size=%d per_gpu_batch=%d global_batch=%d epochs=%d activation_checkpoint=%s",
                        run_dir, runtime.world_size, config["training"]["batch_size"],
                        runtime.world_size * config["training"]["batch_size"],
                        config["training"]["epochs"],
                        args.activation_checkpoint)
            logger.info("train/val/test=%d/%d/%d; training padding=%d samples per epoch",
                        ntrain, nval, ntest, padded_count - ntrain)

        random.seed(args.seed + runtime.rank)
        np.random.seed(args.seed + runtime.rank)
        torch.manual_seed(args.seed + runtime.rank)
        if runtime.device.type == "cuda":
            torch.cuda.manual_seed(args.seed + runtime.rank)

        if checkpoint is None:
            mean = std = None
            if runtime.primary:
                logger.info("Computing training-only ocean U/V statistics from H5 on rank 0")
                mean, std = compute_train_uv_stats(dataset, ntrain)
                atomic_save({"mean": mean, "std": std, "train_samples": ntrain,
                             "input_days": dataset.input_days}, run_dir / "uv_stats.pt")
            epoch = local_offset = global_step = 0
            best_val = float("inf")
            test_loss = None
        else:
            mean, std = checkpoint["uv_mean"], checkpoint["uv_std"]
            epoch = checkpoint["epoch"]
            local_offset = checkpoint["local_offset"]
            global_step = checkpoint["global_step"]
            best_val = checkpoint["best_val"]
            test_loss = checkpoint["test_loss"]
            if runtime.primary:
                logger.info("Resuming DDP epoch=%d local_offset=%d global_step=%d",
                            epoch, local_offset, global_step)

        uv_mean, uv_std = runtime.broadcast_stats(mean, std)
        model = Model(dict2namespace(config["model"]),
                      activation_checkpoint=args.activation_checkpoint).to(runtime.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config["training"]["scheduler_step"],
            gamma=config["training"]["scheduler_gamma"])
        loss_fn = MaskedLpLoss(reduction=False)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None,
            output_device=runtime.local_rank if runtime.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            runtime.restore_rng_state(checkpoint["rng_states"][runtime.rank])

        def write_checkpoint():
            save_latest(runtime, latest_path, model, optimizer, scheduler, config, args,
                        h5_path, splits, per_rank, uv_mean, uv_std, epoch, local_offset,
                        global_step, best_val, test_loss)

        if checkpoint is None:
            write_checkpoint()

        stop_requested = {"value": False}

        def request_stop(_signum, _frame):
            stop_requested["value"] = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        steps_this_run = 0
        batch_size = config["training"]["batch_size"]
        best_path = run_dir / "checkpoint_best.pth"

        while epoch < config["training"]["epochs"]:
            ddp_model.train()
            started = perf_counter()
            indices = train_indices(ntrain, runtime.world_size, runtime.rank,
                                    args.seed, epoch, local_offset)
            train_loader = make_loader(dataset, indices, batch_size, args.num_workers,
                                       runtime.device, args.seed + epoch + runtime.rank * 1000000)
            for batch in train_loader:
                x, y, valid, positions = move_batch(batch, runtime.device, uv_mean, uv_std)
                prediction = ddp_model(x, positions) * uv_std + uv_mean
                losses = loss_fn(prediction, y, valid)
                valid_samples = runtime.sum_int(valid.flatten(1).any(dim=1).sum().item())
                # DDP averages gradients across ranks; undo that factor so the
                # update is the mean over all valid ocean samples globally.
                loss = losses.sum() * (runtime.world_size / max(valid_samples, 1))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite DDP loss at epoch {epoch}, step {global_step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if valid_samples:
                    optimizer.step()
                local_offset += x.shape[0]
                global_step += 1
                steps_this_run += 1
                if global_step % args.log_every_steps == 0:
                    mean_loss = runtime.mean(loss.item())
                    if runtime.primary:
                        logger.info("epoch=%d samples=%d/%d step=%d loss=%.6f",
                                    epoch, min(local_offset * runtime.world_size, ntrain),
                                    ntrain, global_step, mean_loss)
                should_stop = runtime.any_true(stop_requested["value"] or (
                    args.stop_after_steps is not None and steps_this_run >= args.stop_after_steps))
                if global_step % args.checkpoint_every_steps == 0 or should_stop:
                    write_checkpoint()
                    if runtime.primary:
                        logger.info("DDP checkpoint saved at epoch=%d local_offset=%d/%d",
                                    epoch, local_offset, per_rank)
                if should_stop:
                    if runtime.primary:
                        logger.info("Stopped safely; resume with --resume %s", latest_path)
                    return 0

            # Resume at this boundary by repeating validation, never optimizer steps.
            local_offset = per_rank
            write_checkpoint()
            val_indices = evaluation_indices(ntrain, nval, runtime.world_size, runtime.rank)
            val_loader = make_loader(dataset, val_indices, batch_size, args.num_workers,
                                     runtime.device, args.seed + 1000000 + epoch + runtime.rank)
            val_loss = evaluate_distributed(model, val_loader, loss_fn, runtime,
                                            uv_mean, uv_std, stop_requested)
            if val_loss is None:
                if runtime.primary:
                    logger.info("Validation interrupted; resume with --resume %s", latest_path)
                return 0
            if val_loss < best_val:
                best_val = val_loss
                if runtime.primary:
                    atomic_save({"model": model.state_dict(), "epoch": epoch,
                                 "val_loss": val_loss}, best_path)
            scheduler.step()
            if runtime.primary:
                logger.info("epoch=%d complete seconds=%.1f val_L2=%.6f best_val_L2=%.6f",
                            epoch, perf_counter() - started, val_loss, best_val)
            epoch += 1
            local_offset = 0
            write_checkpoint()

        if test_loss is None:
            runtime.barrier()
            best = torch.load(best_path, map_location=runtime.device, weights_only=True)
            final_state = {name: value.detach().cpu().clone()
                           for name, value in model.state_dict().items()}
            model.load_state_dict(best["model"])
            test_indices = evaluation_indices(ntrain + nval, ntest, runtime.world_size, runtime.rank)
            test_loader = make_loader(dataset, test_indices, batch_size, args.num_workers,
                                      runtime.device, args.seed + 2000000 + runtime.rank)
            test_loss = evaluate_distributed(model, test_loader, loss_fn, runtime,
                                             uv_mean, uv_std, stop_requested)
            if test_loss is None:
                if runtime.primary:
                    logger.info("Test interrupted; resume with --resume %s", latest_path)
                return 0
            if runtime.primary:
                logger.info("test_L2=%.6f best_epoch=%d", test_loss, best["epoch"])
            model.load_state_dict(final_state)
            write_checkpoint()
        elif runtime.primary:
            logger.info("Run already completed: test_L2=%.6f", test_loss)
        return 0
    finally:
        dataset.close()
