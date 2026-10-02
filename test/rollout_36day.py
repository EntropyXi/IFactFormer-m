"""Free-run the trained 7-to-1 model across the 36-day held-out test period.

Only the seven days immediately before the test period seed each patch. Every
later input frame contains the preceding *prediction*, never a future observed
velocity. Future H5 frames are read only after prediction, for scoring.

Example:
    python -u test/rollout_36day.py --run-dir results/your_run --device cuda:0

Outputs: predictions.h5 [lead_day, patch, U/V, H, W], per_patch_day.csv,
per_lead.csv, and summary.json. No full source H5 is loaded into memory.
"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parent
for directory in (str(ROOT), str(TEST_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from evaluate_one_step import add_pixel_errors, finalize_pixel_errors, load_run
from libs.copernicus_h5_dataset import CopernicusH5Dataset, build_patch_positions
from libs.training_common import split_counts
from libs.utils import MaskedLpLoss, normalize_uv_input


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--h5", type=Path, default=None,
                        help="Defaults to the H5 path recorded in checkpoint_latest.pt")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Must not already exist; default is next rollout_36day_XXXX")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon-days", type=int, default=36,
                        help="Use fewer days only for a labelled partial smoke test")
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Use fewer patches only for a labelled partial smoke test")
    return parser.parse_args()


def new_output_directory(run_dir, explicit):
    if explicit is not None:
        output_dir = explicit.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir
    for index in range(10000):
        output_dir = run_dir / f"rollout_36day_{index:04d}"
        try:
            output_dir.mkdir()
            return output_dir
        except FileExistsError:
            continue
    raise RuntimeError("No unused rollout directory was found")


def load_seed(h5_file, first_input_day, patch, patches_per_day, input_days, device,
              bounds=None):
    """Read only pre-test input days and their first-day coordinates."""
    rows = first_input_day * patches_per_day + patch + np.arange(input_days) * patches_per_day
    y0, y1, x0, x1 = bounds if bounds is not None else (0, None, 0, None)
    ys, xs = slice(y0, y1), slice(x0, x1)
    uv = np.asarray(h5_file["uovo_data"][rows, :, ys, xs], dtype=np.float32)
    land = np.asarray(h5_file["mask"][rows, ys, xs], dtype=np.bool_)
    valid = (~land) & np.isfinite(uv).all(axis=1)
    clean_uv = np.where(valid[:, None], uv, 0.0)
    history = np.concatenate((np.moveaxis(clean_uv, 1, -1),
                              valid[..., None].astype(np.float32)), axis=-1)
    positions = build_patch_positions(h5_file["lat"][int(rows[0]), ys, xs],
                                      h5_file["lon"][int(rows[0]), ys, xs])
    positions = {key: value.unsqueeze(0).to(device) for key, value in positions.items()}
    return (torch.from_numpy(history.copy()).unsqueeze(0).to(device),
            positions, valid[-1])


def read_target(h5_file, target_day, patch, patches_per_day, device):
    """Read one future frame only for scoring, after its forecast is produced."""
    row = target_day * patches_per_day + patch
    uv = np.asarray(h5_file["uovo_data"][row], dtype=np.float32)
    land = np.asarray(h5_file["mask"][row], dtype=np.bool_)
    valid = (~land) & np.isfinite(uv).all(axis=0)
    clean_uv = np.where(valid[None], uv, 0.0)
    target = torch.from_numpy(np.moveaxis(clean_uv, 0, -1).copy()).unsqueeze(0).to(device)
    return target, torch.from_numpy(valid.copy()).unsqueeze(0).to(device)


def iter_free_rollout(model, history, positions, uv_mean, uv_std, horizon_days):
    """Yield physical U/V; append only previous predictions and the seed mask."""
    if history.ndim != 5 or history.shape[-1] != 3 or horizon_days < 1:
        raise ValueError("Expected history [B, T, H, W, 3] and a positive horizon")
    seed_mask = (history[:, -1, ..., 2:3] > 0.5).to(history.dtype)
    for lead in range(horizon_days):
        normalized = normalize_uv_input(history, uv_mean, uv_std)
        raw_prediction = model(normalized, positions) * uv_std + uv_mean
        if not torch.isfinite(raw_prediction).all():
            raise ValueError(f"Non-finite prediction at lead day {lead + 1}")
        prediction = torch.where(seed_mask.bool(), raw_prediction, 0.0)
        yield prediction
        next_frame = torch.cat((prediction, seed_mask), dim=-1).unsqueeze(1)
        history = torch.cat((history[:, 1:], next_frame), dim=1)


def create_predictions_h5(path, horizon_days, patch_count, spatial_shape,
                          first_target_day, input_days, source_path, best_path,
                          chunk_shape=None):
    height, width = spatial_shape
    chunk_height, chunk_width = chunk_shape or spatial_shape
    output = h5py.File(path, "w")
    output.attrs["status"] = "incomplete"
    output.attrs["completed_patches"] = 0
    output.attrs["free_rollout"] = True
    output.attrs["input_days"] = input_days
    output.attrs["horizon_days"] = horizon_days
    output.attrs["first_target_day_index_zero_based"] = first_target_day
    output.attrs["source_h5"] = str(source_path)
    output.attrs["checkpoint_best"] = str(best_path)
    output.attrs["mask_policy"] = "Reuse last observed valid mask; never read future masks as model input"
    predictions = output.create_dataset(
        "pred_uv", shape=(horizon_days, patch_count, 2, height, width), dtype="f4",
        chunks=(1, 1, 2, chunk_height, chunk_width), compression="lzf", shuffle=True,
        fillvalue=np.nan)
    seed_masks = output.create_dataset(
        "seed_ocean_mask", shape=(patch_count, height, width), dtype="bool",
        chunks=(1, chunk_height, chunk_width), compression="lzf")
    output.create_dataset("target_day_indices", data=np.arange(
        first_target_day, first_target_day + horizon_days, dtype=np.int32))
    output.create_dataset("patch_indices", data=np.arange(patch_count, dtype=np.int32))
    return output, predictions, seed_masks


def evaluate(args):
    if args.horizon_days < 1 or (args.max_patches is not None and args.max_patches < 1):
        raise ValueError("horizon-days and max-patches must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    run_dir = args.run_dir.resolve()
    latest, best, model, mean, std, latest_path, best_path = load_run(run_dir, device)
    h5_path = (args.h5 if args.h5 is not None else Path(latest["h5_path"])).resolve()
    input_days = int(latest["model_config"]["in_time_window"])
    patches_per_day = int(latest["patches_per_day"])
    dataset = CopernicusH5Dataset(h5_path, patches_per_day=patches_per_day,
                                  input_days=input_days,
                                  window_size=latest.get("window_size"))
    try:
        splits = split_counts(dataset)
        if tuple(latest["splits"]) != splits:
            raise ValueError(f"H5 split {splits} differs from checkpoint {latest['splits']}")
        ntrain, nval, ntest = splits
        if (ntrain + nval) % dataset.samples_per_day or ntest % dataset.samples_per_day:
            raise ValueError("Test split must begin and end on day boundaries")
        test_days = ntest // dataset.samples_per_day
        if args.horizon_days > test_days:
            raise ValueError(f"Horizon {args.horizon_days} exceeds {test_days} held-out days")
        patch_count = min(patches_per_day, args.max_patches or patches_per_day)
        first_input_day = (ntrain + nval) // dataset.samples_per_day
        first_target_day = first_input_day + input_days
        if first_target_day + args.horizon_days > dataset.num_days:
            raise ValueError("Forecast horizon extends beyond the H5 time axis")
        output_dir = new_output_directory(run_dir, args.output_dir)
        horizon = args.horizon_days
        loss_fn = MaskedLpLoss(reduction=False)
        loss_sums = np.zeros(horizon, dtype=np.float64)
        target_pixel_counts = np.zeros(horizon, dtype=np.int64)
        covered_pixel_counts = np.zeros(horizon, dtype=np.int64)
        pixel_totals = [defaultdict(float) for _ in range(horizon)]
        fields = ("lead_day", "target_day_index", "patch_index", "masked_relative_l2",
                  "valid_ocean_pixels", "valid_pixels_in_seed_mask")
        with (h5py.File(h5_path, "r") as source,
              (output_dir / "per_patch_day.csv").open("w", newline="", encoding="utf-8") as detail_file):
            writer = csv.DictWriter(detail_file, fieldnames=fields)
            writer.writeheader()
            output_h5, predictions, seed_masks = create_predictions_h5(
                output_dir / "predictions.h5", horizon, patch_count,
                dataset.source_spatial_shape,
                first_target_day, input_days, h5_path, best_path,
                chunk_shape=dataset.spatial_shape)
            with output_h5, torch.inference_mode():
                for patch in range(patch_count):
                    for tile_row in range(dataset.window_rows):
                        for tile_col in range(dataset.window_cols):
                            y0, y1, x0, x1 = dataset.window_bounds(tile_row, tile_col)
                            history, positions, seed_valid = load_seed(
                                source, first_input_day, patch, patches_per_day,
                                input_days, device, bounds=(y0, y1, x0, x1))
                            seed_masks[patch, y0:y1, x0:x1] = seed_valid
                            for lead_index, prediction in enumerate(
                                    iter_free_rollout(model, history, positions,
                                                      mean, std, horizon)):
                                predictions[lead_index, patch, :, y0:y1, x0:x1] = np.moveaxis(
                                    prediction[0].cpu().numpy(), -1, 0)

                    seed_valid_device = torch.from_numpy(seed_masks[patch]).unsqueeze(0).to(device)
                    for lead_index in range(horizon):
                        prediction = torch.from_numpy(np.moveaxis(
                            predictions[lead_index, patch], 0, -1).copy()).unsqueeze(0).to(device)
                        target_day = first_target_day + lead_index
                        # This read occurs after prediction; target never enters history.
                        target, target_valid = read_target(
                            source, target_day, patch, patches_per_day, device)
                        relative_l2 = loss_fn(prediction, target, target_valid).item()
                        loss_sums[lead_index] += relative_l2
                        add_pixel_errors(pixel_totals[lead_index], prediction, target,
                                         target_valid)
                        valid_count = int(target_valid.sum().item())
                        covered_count = int((target_valid & seed_valid_device).sum().item())
                        target_pixel_counts[lead_index] += valid_count
                        covered_pixel_counts[lead_index] += covered_count
                        writer.writerow({
                            "lead_day": lead_index + 1,
                            "target_day_index": target_day,
                            "patch_index": patch,
                            "masked_relative_l2": relative_l2,
                            "valid_ocean_pixels": valid_count,
                            "valid_pixels_in_seed_mask": covered_count,
                        })
                    output_h5.attrs["completed_patches"] = patch + 1
                    output_h5.flush()
                    detail_file.flush()
                    if (patch + 1) % 5 == 0 or patch + 1 == patch_count:
                        print(f"Completed {patch + 1}/{patch_count} patches "
                              f"({horizon} free-rollout days each)", flush=True)
                output_h5.attrs["status"] = "complete"
                output_h5.flush()

        lead_fields = ("lead_day", "target_day_index", "sample_mean_masked_relative_l2",
                       "valid_ocean_pixels", "target_pixel_coverage_by_seed_mask",
                       "u_mae", "u_rmse", "v_mae", "v_rmse", "speed_mae", "speed_rmse")
        per_lead = []
        with (output_dir / "per_lead.csv").open("w", newline="", encoding="utf-8") as lead_file:
            writer = csv.DictWriter(lead_file, fieldnames=lead_fields)
            writer.writeheader()
            for index in range(horizon):
                metrics = finalize_pixel_errors(pixel_totals[index])
                row = {
                    "lead_day": index + 1,
                    "target_day_index": first_target_day + index,
                    "sample_mean_masked_relative_l2": loss_sums[index] / patch_count,
                    "valid_ocean_pixels": int(target_pixel_counts[index]),
                    "target_pixel_coverage_by_seed_mask": (covered_pixel_counts[index] /
                                                          target_pixel_counts[index]),
                    "u_mae": metrics["u"]["mae"], "u_rmse": metrics["u"]["rmse"],
                    "v_mae": metrics["v"]["mae"], "v_rmse": metrics["v"]["rmse"],
                    "speed_mae": metrics["speed"]["mae"],
                    "speed_rmse": metrics["speed"]["rmse"],
                }
                writer.writerow(row)
                per_lead.append(row)

        summary = {
            "source_h5": str(h5_path),
            "checkpoint_latest": str(latest_path),
            "checkpoint_best": str(best_path),
            "best_epoch_zero_based": int(best["epoch"]) if "epoch" in best else None,
            "mode": "free_rollout_no_future_velocity_or_mask_as_input",
            "input_days": input_days,
            "window_size": dataset.window_size,
            "windows_per_patch": dataset.tiles_per_patch,
            "horizon_days": horizon,
            "first_input_day_index": first_input_day,
            "first_target_day_index": first_target_day,
            "last_target_day_index": first_target_day + horizon - 1,
            "evaluated_patches": patch_count,
            "total_patches_per_day": patches_per_day,
            "complete_test_horizon": horizon == test_days and patch_count == patches_per_day,
            "split_samples": {"train": ntrain, "validation": nval, "test": ntest},
            "all_leads_sample_mean_masked_relative_l2": float(loss_sums.sum() /
                                                               (horizon * patch_count)),
            "first_lead_sample_mean_masked_relative_l2": float(loss_sums[0] / patch_count),
            "final_lead_sample_mean_masked_relative_l2": float(loss_sums[-1] / patch_count),
            "target_pixel_coverage_by_seed_mask": float(covered_pixel_counts.sum() /
                                                        target_pixel_counts.sum()),
            "prediction_file": str(output_dir / "predictions.h5"),
            "per_lead": per_lead,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({"output_dir": str(output_dir),
                          "complete_test_horizon": summary["complete_test_horizon"],
                          "first_lead_relative_l2": summary[
                              "first_lead_sample_mean_masked_relative_l2"],
                          "final_lead_relative_l2": summary[
                              "final_lead_sample_mean_masked_relative_l2"]}, indent=2),
              flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    evaluate(parse_args())
