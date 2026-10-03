"""DDP-only checkpoints, including one RNG stream per rank."""

from libs.training_common import atomic_save


FORMAT_VERSION = 2


def save_latest(runtime, path, model, optimizer, scheduler, config, args, h5_path,
                splits, samples_per_rank, uv_mean, uv_std, epoch, local_offset,
                global_step, best_val, test_loss):
    rng_states = runtime.gather_rng_states()
    if not runtime.primary:
        return
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "mode": "ddp",
        "world_size": runtime.world_size,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "activation_checkpoint": args.activation_checkpoint,
        "model_config": config["model"],
        "training_config": config["training"],
        "h5_path": str(h5_path),
        "splits": splits,
        "patches_per_day": args.patches_per_day,
        "window_size": args.window_size,
        "tile_selection": args.tile_selection,
        "seed": args.seed,
        "uv_mean": uv_mean.detach().cpu(),
        "uv_std": uv_std.detach().cpu(),
        "samples_per_rank": samples_per_rank,
        "epoch": epoch,
        "local_offset": local_offset,
        "global_step": global_step,
        "best_val": best_val,
        "test_loss": test_loss,
        "rng_states": rng_states,
    }
    atomic_save(checkpoint, path)


def verify_resume(checkpoint, config, args, h5_path, splits, runtime, samples_per_rank):
    if checkpoint.get("format_version") != FORMAT_VERSION or checkpoint.get("mode") != "ddp":
        raise ValueError("Expected a DDP checkpoint; single-GPU checkpoints cannot be resumed as DDP")
    if checkpoint["world_size"] != runtime.world_size:
        raise ValueError("DDP world size differs from the checkpoint")
    if checkpoint["model_config"] != config["model"]:
        raise ValueError("Model config differs from the checkpoint")
    if checkpoint.get("activation_checkpoint", False) != args.activation_checkpoint:
        raise ValueError("Activation checkpoint setting differs from the checkpoint")
    for name in ("batch_size", "lr", "scheduler_step", "scheduler_gamma"):
        if checkpoint["training_config"][name] != config["training"][name]:
            raise ValueError(f"Training setting {name} differs from the checkpoint")
    if checkpoint["h5_path"] != str(h5_path) or tuple(checkpoint["splits"]) != splits:
        raise ValueError("Dataset path or split differs from the checkpoint")
    if checkpoint["patches_per_day"] != args.patches_per_day or checkpoint["seed"] != args.seed:
        raise ValueError("Patch count or seed differs from the checkpoint")
    if checkpoint.get("window_size") != args.window_size:
        raise ValueError("Window size differs from the checkpoint")
    if checkpoint.get("tile_selection", "all") != args.tile_selection:
        raise ValueError("Tile selection differs from the checkpoint")
    if checkpoint["samples_per_rank"] != samples_per_rank:
        raise ValueError("Training sample count per rank differs from the checkpoint")
    if not 0 <= checkpoint["local_offset"] <= samples_per_rank:
        raise ValueError("Invalid local sample offset in checkpoint")
    if checkpoint["epoch"] > config["training"]["epochs"]:
        raise ValueError("Configured epochs are fewer than completed epochs")
    if len(checkpoint["rng_states"]) != runtime.world_size:
        raise ValueError("Per-rank RNG state count differs from world size")
