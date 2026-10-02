"""Evaluate the best checkpoint on held-out, one-day-ahead Copernicus windows.

Example:
    python -u test/evaluate_one_step.py --run-dir results/your_run \
        --h5 /data/copernicus_uv_data/processed_data/uovo_mid_1997-01-01_to_1997-12-31.h5

The H5 dataset is read on demand. This script never retrains or rolls predictions
forward; each forecast uses the seven observed days of its own test window.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.copernicus_h5_dataset import CopernicusH5Dataset
from libs.model import Model
from libs.training_common import make_loader, split_counts
from libs.utils import MaskedLpLoss, dict2namespace, normalize_uv_input


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Training directory containing checkpoint_latest.pt and checkpoint_best.pth")
    parser.add_argument("--h5", type=Path, default=None,
                        help="H5 path; defaults to the path recorded in the latest checkpoint")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New output directory; defaults to the next one_step_eval_XXXX under run-dir")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--plot-count", type=int, default=3,
                        help="Number of representative test patches to plot; 0 disables plots")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional smoke-test limit; results will be labelled partial")
    return parser.parse_args()


def new_output_directory(run_dir, explicit):
    if explicit is not None:
        output_dir = explicit.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir
    for index in range(10000):
        output_dir = run_dir / f"one_step_eval_{index:04d}"
        try:
            output_dir.mkdir()
            return output_dir
        except FileExistsError:
            continue
    raise RuntimeError("No unused one-step evaluation directory was found")


def load_run(run_dir, device):
    latest_path = run_dir / "checkpoint_latest.pt"
    best_path = run_dir / "checkpoint_best.pth"
    latest = torch.load(latest_path, map_location="cpu", weights_only=True)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    required = ("model_config", "h5_path", "patches_per_day", "splits", "uv_mean", "uv_std")
    missing = [name for name in required if name not in latest]
    if missing:
        raise ValueError(f"Latest checkpoint is missing metadata: {missing}")
    if "model" not in best:
        raise ValueError("Best checkpoint has no model state")
    model = Model(dict2namespace(latest["model_config"])).to(device)
    model.load_state_dict(best["model"], strict=True)
    model.eval()
    mean = torch.as_tensor(latest["uv_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(latest["uv_std"], dtype=torch.float32, device=device)
    if mean.shape != (2,) or std.shape != (2,) or not torch.isfinite(mean).all() or not (std > 0).all():
        raise ValueError("Checkpoint U/V normalization statistics are invalid")
    return latest, best, model, mean, std, latest_path, best_path


def add_pixel_errors(totals, prediction, target, valid):
    """Accumulate pixel-weighted physical-unit errors without saving full maps."""
    mask = valid.unsqueeze(-1)
    difference = torch.where(mask, prediction - target, 0)
    reference = torch.where(mask, target, 0)
    totals["pixels"] += int(valid.sum().item())
    for channel, name in enumerate(("u", "v")):
        error = difference[..., channel]
        totals[f"{name}_abs"] += error.abs().sum().item()
        totals[f"{name}_sq"] += error.square().sum().item()
        totals[f"{name}_target_sq"] += reference[..., channel].square().sum().item()
    pred_speed = torch.linalg.vector_norm(prediction, dim=-1)
    true_speed = torch.linalg.vector_norm(target, dim=-1)
    speed_error = torch.where(valid, pred_speed - true_speed, 0)
    totals["speed_abs"] += speed_error.abs().sum().item()
    totals["speed_sq"] += speed_error.square().sum().item()
    totals["speed_target_sq"] += torch.where(valid, true_speed.square(), 0).sum().item()


def finalize_pixel_errors(totals):
    pixels = totals["pixels"]
    if pixels == 0:
        raise ValueError("Evaluation contains no valid ocean pixels")
    result = {"valid_ocean_pixels": pixels}
    for name in ("u", "v", "speed"):
        result[name] = {
            "mae": totals[f"{name}_abs"] / pixels,
            "rmse": (totals[f"{name}_sq"] / pixels) ** 0.5,
            "global_relative_l2": (totals[f"{name}_sq"] /
                                   max(totals[f"{name}_target_sq"], 1e-16)) ** 0.5,
        }
    return result


def plot_patch(path, target, prediction, valid, sample_index, target_day, patch_index,
               tile_row=0, tile_col=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = target.numpy()
    prediction = prediction.numpy()
    valid = valid.numpy().astype(bool)
    target_maps = (target[..., 0], target[..., 1], np.linalg.norm(target, axis=-1))
    prediction_maps = (prediction[..., 0], prediction[..., 1], np.linalg.norm(prediction, axis=-1))
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, label in enumerate(("U", "V", "speed")):
        observed = np.ma.array(target_maps[row], mask=~valid)
        forecast = np.ma.array(prediction_maps[row], mask=~valid)
        error = np.ma.array(np.abs(prediction_maps[row] - target_maps[row]), mask=~valid)
        low = min(observed.min(), forecast.min())
        high = max(observed.max(), forecast.max())
        for col, (data, title) in enumerate(((observed, "truth"), (forecast, "prediction"),
                                             (error, "absolute error"))):
            image = axes[row, col].imshow(data, origin="lower",
                                          cmap="magma" if col == 2 else "coolwarm",
                                          vmin=0 if col == 2 else low,
                                          vmax=None if col == 2 else high)
            axes[row, col].set_title(f"{label}: {title}")
            fig.colorbar(image, ax=axes[row, col], shrink=0.75)
    fig.suptitle(f"Test sample {sample_index} | target day {target_day} | "
                 f"patch {patch_index} | tile ({tile_row}, {tile_col})")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate(args):
    if args.batch_size < 1 or args.num_workers < 0 or args.plot_count < 0:
        raise ValueError("batch-size must be positive; num-workers and plot-count cannot be negative")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for a CPU evaluation")
    run_dir = args.run_dir.resolve()
    latest, best, model, mean, std, latest_path, best_path = load_run(run_dir, device)
    h5_path = (args.h5 if args.h5 is not None else Path(latest["h5_path"])).resolve()
    dataset = CopernicusH5Dataset(
        h5_path, patches_per_day=int(latest["patches_per_day"]),
        input_days=int(latest["model_config"]["in_time_window"]),
        window_size=latest.get("window_size"))
    try:
        splits = split_counts(dataset)
        if tuple(latest["splits"]) != splits:
            raise ValueError(f"H5 split {splits} does not match training checkpoint {latest['splits']}")
        ntrain, nval, ntest = splits
        sample_count = min(ntest, args.max_samples) if args.max_samples is not None else ntest
        test_indices = range(ntrain + nval, ntrain + nval + sample_count)
        loader = make_loader(dataset, test_indices, args.batch_size, args.num_workers, device, seed=0)
        plot_count = min(sample_count, args.plot_count)
        plot_indices = set(np.linspace(0, sample_count - 1, plot_count, dtype=int).tolist())
        output_dir = new_output_directory(run_dir, args.output_dir)

        model_loss_sum = 0.0
        valid_samples = 0
        common_model_loss_sum = 0.0
        persistence_loss_sum = 0.0
        common_samples = 0
        target_pixels = 0
        model_pixels = defaultdict(float)
        model_common_pixels = defaultdict(float)
        persistence_pixels = defaultdict(float)
        relative_loss = MaskedLpLoss(reduction=False)
        processed = 0
        fields = ("test_sample_index", "dataset_index", "target_day_index", "patch_index",
                  "tile_row", "tile_col", "window_y0", "window_x0",
                  "valid_ocean_pixels", "common_ocean_pixels", "model_relative_l2",
                  "model_relative_l2_common", "persistence_relative_l2_common")
        with (output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            with torch.inference_mode():
                for x_raw, target, valid, positions in loader:
                    x_raw = x_raw.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    valid = valid.to(device, non_blocking=True).bool()
                    positions = {key: value.to(device, non_blocking=True)
                                 for key, value in positions.items()}
                    normalized = normalize_uv_input(x_raw, mean, std)
                    prediction = model(normalized, positions) * std + mean
                    model_losses = relative_loss(prediction, target, valid)
                    model_loss_sum += model_losses.sum().item()
                    valid_samples += int(valid.flatten(1).any(dim=1).sum().item())
                    add_pixel_errors(model_pixels, prediction, target, valid)

                    previous_valid = x_raw[:, -1, ..., 2].bool()
                    common_valid = valid & previous_valid
                    persistence = x_raw[:, -1, ..., :2]
                    target_pixels += int(valid.sum().item())
                    for local_index in range(target.shape[0]):
                        sample_index = processed + local_index
                        dataset_index = ntrain + nval + sample_index
                        first_day, patch, tile_row, tile_col = dataset.decode_index(dataset_index)
                        y0, _, x0, _ = dataset.window_bounds(tile_row, tile_col)
                        has_ocean = bool(valid[local_index].any().item())
                        row = {
                            "test_sample_index": sample_index,
                            "dataset_index": dataset_index,
                            "target_day_index": first_day + dataset.input_days,
                            "patch_index": patch,
                            "tile_row": tile_row,
                            "tile_col": tile_col,
                            "window_y0": y0,
                            "window_x0": x0,
                            "valid_ocean_pixels": int(valid[local_index].sum().item()),
                            "common_ocean_pixels": int(common_valid[local_index].sum().item()),
                            "model_relative_l2": (float(model_losses[local_index].item())
                                                  if has_ocean else ""),
                            "model_relative_l2_common": "",
                            "persistence_relative_l2_common": "",
                        }
                        if common_valid[local_index].any():
                            common = common_valid[local_index:local_index + 1]
                            truth = target[local_index:local_index + 1]
                            forecast = prediction[local_index:local_index + 1]
                            previous = persistence[local_index:local_index + 1]
                            model_common_loss = relative_loss(forecast, truth, common).item()
                            persistence_loss = relative_loss(previous, truth, common).item()
                            row["model_relative_l2_common"] = model_common_loss
                            row["persistence_relative_l2_common"] = persistence_loss
                            common_model_loss_sum += model_common_loss
                            persistence_loss_sum += persistence_loss
                            common_samples += 1
                            add_pixel_errors(model_common_pixels, forecast, truth, common)
                            add_pixel_errors(persistence_pixels, previous, truth, common)
                        writer.writerow(row)
                        if sample_index in plot_indices and has_ocean:
                            plot_patch(output_dir / f"sample_{sample_index:05d}.png",
                                       target[local_index].cpu(), prediction[local_index].cpu(),
                                       valid[local_index].cpu(), sample_index,
                                       row["target_day_index"], patch, tile_row, tile_col)
                    processed += target.shape[0]
                    if processed % 500 == 0 or processed == sample_count:
                        print(f"Evaluated {processed}/{sample_count} test samples", flush=True)

        if processed != sample_count:
            raise RuntimeError(f"Expected {sample_count} samples, received {processed}")
        if valid_samples == 0:
            raise ValueError("Evaluation contains no valid ocean samples")
        common_pixel_count = model_common_pixels["pixels"]
        summary = {
            "h5_path": str(h5_path),
            "checkpoint_latest": str(latest_path),
            "checkpoint_best": str(best_path),
            "best_epoch_zero_based": int(best["epoch"]) if "epoch" in best else None,
            "input_days": dataset.input_days,
            "forecast_days": 1,
            "window_size": dataset.window_size,
            "split_samples": {"train": ntrain, "validation": nval, "test": ntest},
            "evaluated_test_samples": processed,
            "valid_ocean_samples": valid_samples,
            "complete_test_set": processed == ntest,
            "model": {
                "sample_mean_masked_relative_l2": model_loss_sum / valid_samples,
                "target_ocean_pixel_metrics": finalize_pixel_errors(model_pixels),
            },
            "persistence_comparison": {
                "definition": "Last observed day's U/V; both methods scored on target/previous valid intersection",
                "comparable_samples": common_samples,
                "common_ocean_pixels": common_pixel_count,
                "target_ocean_pixel_coverage": common_pixel_count / target_pixels,
                "model_sample_mean_masked_relative_l2": (common_model_loss_sum / common_samples
                                                         if common_samples else None),
                "persistence_sample_mean_masked_relative_l2": (persistence_loss_sum / common_samples
                                                               if common_samples else None),
                "model_common_pixel_metrics": (finalize_pixel_errors(model_common_pixels)
                                               if common_samples else None),
                "persistence_common_pixel_metrics": (finalize_pixel_errors(persistence_pixels)
                                                     if common_samples else None),
            },
            "training_recorded_test_relative_l2": latest.get("test_loss"),
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({"output_dir": str(output_dir),
                          "model_test_relative_l2": summary["model"]["sample_mean_masked_relative_l2"],
                          "persistence_relative_l2_on_common_mask": summary["persistence_comparison"][
                              "persistence_sample_mean_masked_relative_l2"]}, indent=2), flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    evaluate(parse_args())
