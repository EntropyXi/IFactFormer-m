"""Plot comparable speed/error maps and four-point 36-day error trajectories.

Uses the finished predictions.h5 and the original H5 as read-only inputs.
The persistence reference repeats the last observed U/V field for all leads.

Example:
    python -u test/plot_rollout_figures.py \
        --rollout-dir results/your_run/rollout_36day_full_20260930
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DAYS = (1, 5, 10, 20, 30)
POINT_LABELS = ("A", "B", "C", "D")
POINT_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, default=None,
                        help="Defaults to source_h5 recorded in rollout summary.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New directory; default is figures_high_ocean_patch_XXX")
    parser.add_argument("--patch", type=int, default=None,
                        help="Default: patch with highest initial ocean fraction")
    parser.add_argument("--days", type=int, nargs=5, default=DEFAULT_DAYS)
    parser.add_argument("--relative-error-cap", type=float, default=3.0,
                        help="Shared map colorbar ceiling; values above it are shown saturated")
    parser.add_argument("--denominator-floor", type=float, default=1e-6,
                        help="Floor for pointwise ||prediction-GT|| / ||GT||")
    return parser.parse_args()


def choose_points(seed_mask):
    """Pick four spatially separated seed-ocean pixels without looking at targets."""
    height, width = seed_mask.shape
    candidates = np.argwhere(seed_mask)
    if candidates.shape[0] < 4:
        raise ValueError("Need at least four ocean pixels for point trajectories")
    centers = ((height // 4, width // 4), (height // 4, 3 * width // 4),
               (3 * height // 4, width // 4), (3 * height // 4, 3 * width // 4))
    chosen = []
    for label, (center_y, center_x) in zip(POINT_LABELS, centers):
        distance = ((candidates[:, 0] - center_y) ** 2 +
                    (candidates[:, 1] - center_x) ** 2)
        for index in np.argsort(distance):
            row, col = map(int, candidates[index])
            if (row, col) not in {(point["row"], point["col"]) for point in chosen}:
                chosen.append({"id": label, "row": row, "col": col,
                               "seed_ocean": True})
                break
    return chosen


def vector_relative_error(predicted_uv, true_uv, denominator_floor):
    """Pointwise vector relative error, not relative speed-magnitude error."""
    numerator = np.linalg.norm(predicted_uv - true_uv, axis=0)
    denominator = np.maximum(np.linalg.norm(true_uv, axis=0), denominator_floor)
    return numerator / denominator


def load_patch(source_h5, prediction_h5, patch, first_target_day, patches_per_day):
    prediction = np.asarray(prediction_h5["pred_uv"][:, patch], dtype=np.float32)
    horizon, channels, height, width = prediction.shape
    if channels != 2:
        raise ValueError("Expected two predicted velocity channels")
    seed_row = (first_target_day - 1) * patches_per_day + patch
    baseline = np.asarray(source_h5["uovo_data"][seed_row], dtype=np.float32)
    seed_valid = ((~source_h5["mask"][seed_row]) & np.isfinite(baseline).all(axis=0))
    recorded_mask = prediction_h5["seed_ocean_mask"][patch]
    if not np.array_equal(seed_valid, recorded_mask):
        raise ValueError("Rollout seed mask differs from source H5")
    baseline = np.where(seed_valid[None], baseline, 0.0)
    truth = np.empty_like(prediction)
    common_valid = np.empty((horizon, height, width), dtype=np.bool_)
    target_valid_count = 0
    common_valid_count = 0
    for index in range(horizon):
        row = (first_target_day + index) * patches_per_day + patch
        frame = np.asarray(source_h5["uovo_data"][row], dtype=np.float32)
        target_valid = ((~source_h5["mask"][row]) & np.isfinite(frame).all(axis=0))
        common_valid[index] = target_valid & seed_valid
        target_valid_count += int(target_valid.sum())
        common_valid_count += int(common_valid[index].sum())
        truth[index] = np.where(target_valid[None], frame, 0.0)
    return prediction, truth, baseline, seed_valid, common_valid, {
        "target_valid_pixels": target_valid_count,
        "common_valid_pixels": common_valid_count,
        "common_mask_coverage": common_valid_count / target_valid_count,
    }


def derive_maps_and_points(prediction, truth, baseline, valid, points, days,
                           denominator_floor, speed_percentile=99.5,
                           relative_error_cap=3.0):
    horizon = prediction.shape[0]
    if any(day < 1 or day > horizon for day in days):
        raise ValueError(f"All shown days must be in 1..{horizon}")
    selected = []
    all_speeds = []
    relative_clipping = {"IFactFormer-m": [0, 0], "Persistence": [0, 0]}
    floor_affected = 0
    floor_total = 0
    for day in days:
        index = day - 1
        ground_truth = truth[index]
        model_uv = prediction[index]
        mask = valid[index]
        true_speed = np.linalg.norm(ground_truth, axis=0)
        model_speed = np.linalg.norm(model_uv, axis=0)
        baseline_speed = np.linalg.norm(baseline, axis=0)
        model_error = vector_relative_error(model_uv, ground_truth, denominator_floor)
        baseline_error = vector_relative_error(baseline, ground_truth,
                                               denominator_floor)
        selected.append({"day": day, "valid": mask,
                         "speeds": (true_speed, model_speed, baseline_speed),
                         "errors": (np.zeros_like(true_speed), model_error,
                                    baseline_error)})
        all_speeds.extend((item[mask] for item in (true_speed, model_speed,
                                                   baseline_speed)))
        floor_affected += int(np.count_nonzero((true_speed < denominator_floor) & mask))
        floor_total += int(mask.sum())
        for name, error in (("IFactFormer-m", model_error),
                            ("Persistence", baseline_error)):
            relative_clipping[name][0] += int(np.count_nonzero(error[mask] > relative_error_cap))
            relative_clipping[name][1] += int(mask.sum())
    speed_values = np.concatenate(all_speeds)
    speed_cap = float(np.percentile(speed_values, speed_percentile))
    speed_clipped = int(np.count_nonzero(speed_values > speed_cap))
    point_rows = []
    for day in range(1, horizon + 1):
        for point in points:
            row, col = point["row"], point["col"]
            is_valid = bool(valid[day - 1, row, col])
            ground_truth = truth[day - 1, :, row, col]
            model_uv = prediction[day - 1, :, row, col]
            speed = float(np.linalg.norm(ground_truth)) if is_valid else None
            denominator = max(speed, denominator_floor) if is_valid else None
            point_rows.append({
                "lead_day": day,
                "point_id": point["id"], "row": row, "col": col,
                "valid": is_valid,
                "truth_speed": speed,
                "model_relative_error": (float(np.linalg.norm(model_uv - ground_truth) /
                                               denominator) if is_valid else None),
                "baseline_relative_error": (float(np.linalg.norm(baseline[:, row, col] -
                                                                  ground_truth) /
                                                  denominator) if is_valid else None),
                "denominator_floored": bool(is_valid and speed < denominator_floor),
            })
    scale_info = {
        "speed_percentile": speed_percentile,
        "speed_colorbar_max": speed_cap,
        "speed_values_above_colorbar": speed_clipped,
        "speed_value_count": int(speed_values.size),
        "relative_error_colorbar_max": relative_error_cap,
        "relative_error_values_above_colorbar": {
            name: {"count": count, "total": total}
            for name, (count, total) in relative_clipping.items()
        },
        "denominator_floor": denominator_floor,
        "map_truth_pixels_using_floor": floor_affected,
        "map_truth_valid_pixels": floor_total,
        "point_truth_values_using_floor": sum(row["denominator_floored"] for row in point_rows),
    }
    return selected, point_rows, scale_info


def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": None,
    })
    return plt


def make_map_figure(plt, selected, patch, ocean_fraction, points, scale_info,
                    kind, output_dir):
    import matplotlib as mpl
    import matplotlib.patheffects as path_effects

    figure = plt.figure(figsize=(7.1, 9.2))
    grid = figure.add_gridspec(5, 4, width_ratios=(1, 1, 1, 0.075),
                               left=0.09, right=0.94, bottom=0.075, top=0.93,
                               wspace=0.10, hspace=0.14)
    axes = np.empty((5, 3), dtype=object)
    if kind == "speed":
        palette = plt.get_cmap("cividis").copy()
        vmax = scale_info["speed_colorbar_max"]
        heading = "Speed magnitude | GT vs IFactFormer-m vs persistence"
        colorbar_label = "Speed (native H5 units)"
        footer = ("One shared 0-to-99.5th-percentile color scale; higher values saturated. "
                  "A-D mark the four trajectory points on GT day +1.")
        filename = "speed_comparison"
    else:
        palette = plt.get_cmap("magma").copy()
        vmax = scale_info["relative_error_colorbar_max"]
        heading = "Pointwise vector relative error | GT reference vs two forecasts"
        colorbar_label = "||pred - GT|| / max(||GT||, 1e-6)"
        footer = ("GT error is zero by definition. Shared color scale; values > 3 saturated. "
                  "Land/invalid pixels shown in gray.")
        filename = "relative_error_comparison"
    palette.set_bad("#E5E7EB")
    names = ("GT" if kind == "speed" else "GT reference (= 0)",
             "IFactFormer-m", "Persistence")
    normalizer = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    for row_index, item in enumerate(selected):
        data = item["speeds"] if kind == "speed" else item["errors"]
        mask = item["valid"]
        for col_index in range(3):
            ax = figure.add_subplot(grid[row_index, col_index])
            axes[row_index, col_index] = ax
            image = ax.imshow(np.ma.array(data[col_index], mask=~mask), origin="lower",
                              interpolation="nearest", cmap=palette, norm=normalizer)
            if row_index == 0:
                ax.set_title(names[col_index], pad=4)
            if col_index == 0:
                ax.set_ylabel(f"Day +{item['day']}\nGrid row")
                ax.set_yticks((0, 224, 447))
            else:
                ax.set_yticks(())
            if row_index == len(selected) - 1:
                ax.set_xlabel("Grid column")
                ax.set_xticks((0, 224, 447))
            else:
                ax.set_xticks(())
            if kind == "speed" and row_index == 0 and col_index == 0:
                for point, color in zip(points, POINT_COLORS):
                    ax.scatter(point["col"], point["row"], s=28, facecolor="white",
                               edgecolor=color, linewidth=1.3, zorder=3)
                    ax.text(point["col"] + 9, point["row"] + 9, point["id"],
                            color=color, fontsize=8, weight="bold", zorder=4,
                            path_effects=[path_effects.withStroke(linewidth=1.6,
                                                                   foreground="white")])
    color_axis = figure.add_subplot(grid[:, 3])
    figure.colorbar(image, cax=color_axis, extend="max", label=colorbar_label)
    figure.suptitle(f"{heading}\nPatch {patch} | seed ocean {ocean_fraction:.2%}", y=0.985)
    figure.text(0.09, 0.025, footer, fontsize=6.5)
    paths = save_pair(figure, output_dir / filename, dpi=300)
    plt.close(figure)
    return paths


def make_point_figure(plt, point_rows, points, patch, ocean_fraction, output_dir):
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.6), sharex=True,
                               gridspec_kw={"left": 0.11, "right": 0.97,
                                            "bottom": 0.13, "top": 0.80,
                                            "wspace": 0.22, "hspace": 0.28})
    for ax, point, color in zip(axes.flat, points, POINT_COLORS):
        rows = [row for row in point_rows if row["point_id"] == point["id"]]
        lead = np.array([row["lead_day"] for row in rows])
        model_error = np.array([row["model_relative_error"] if row["valid"] else np.nan
                                for row in rows])
        baseline_error = np.array([row["baseline_relative_error"] if row["valid"] else np.nan
                                   for row in rows])
        ax.plot(lead, model_error, color=color, linestyle="-", label="IFactFormer-m")
        ax.plot(lead, baseline_error, color="#424242", linestyle="--",
                label="Persistence")
        ax.axhline(1, color="#9E9E9E", linestyle=":", linewidth=0.9)
        ax.set_title(f"{point['id']}  (row {point['row']}, col {point['col']})")
        ax.set_xlim(1, 36)
        ax.set_xticks((1, 5, 10, 20, 30, 36))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)
    axes[1, 0].set_xlabel("Forecast lead (days)")
    axes[1, 1].set_xlabel("Forecast lead (days)")
    axes[0, 0].set_ylabel("Pointwise vector relative error")
    axes[1, 0].set_ylabel("Pointwise vector relative error")
    handles = (Line2D([0], [0], color="#404040", linestyle="-"),
               Line2D([0], [0], color="#424242", linestyle="--"))
    figure.legend(handles, ("IFactFormer-m (point color)", "Persistence"),
                  loc="upper center", bbox_to_anchor=(0.5, 0.875),
                  ncol=2, frameon=False)
    figure.suptitle(f"Four ocean points | 36-day error trajectories\n"
                    f"Patch {patch} | seed ocean {ocean_fraction:.2%}", y=0.99)
    figure.text(0.11, 0.035,
                "Each panel has its own y-scale. Dotted line = error 1 (zero-field reference).",
                fontsize=6.5)
    paths = save_pair(figure, output_dir / "four_point_error_36day", dpi=300)
    plt.close(figure)
    return paths


def save_pair(figure, base, dpi):
    paths = []
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        figure.savefig(path, dpi=dpi if suffix == ".png" else None,
                       bbox_inches=None, pad_inches=0)
        paths.append(str(path))
    return paths


def main(args):
    if args.relative_error_cap <= 0 or args.denominator_floor <= 0:
        raise ValueError("relative-error-cap and denominator-floor must be positive")
    days = tuple(args.days)
    if len(set(days)) != 5 or days != tuple(sorted(days)):
        raise ValueError("--days must contain five distinct increasing lead days")
    rollout_dir = args.rollout_dir.resolve()
    with (rollout_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("complete_test_horizon"):
        raise ValueError("Figures require a complete 36-day rollout")
    source_path = (args.source_h5 or Path(summary["source_h5"])).resolve()
    prediction_path = rollout_dir / "predictions.h5"
    with h5py.File(source_path, "r") as source, h5py.File(prediction_path, "r") as output:
        if output.attrs["status"] != "complete":
            raise ValueError("Prediction H5 is not marked complete")
        horizon, patch_count, channels, height, width = output["pred_uv"].shape
        if horizon != 36 or patch_count != summary["total_patches_per_day"]:
            raise ValueError("Prediction H5 does not contain the complete 36-day test period")
        days = tuple(int(day) for day in days)
        if any(day > horizon for day in days):
            raise ValueError("Requested day exceeds rollout horizon")
        fractions = np.asarray(output["seed_ocean_mask"][:], dtype=np.bool_).mean(axis=(1, 2))
        patch = int(np.argmax(fractions)) if args.patch is None else args.patch
        if not 0 <= patch < patch_count:
            raise ValueError("Patch index is outside the rollout H5")
        ocean_fraction = float(fractions[patch])
        first_target_day = int(output.attrs["first_target_day_index_zero_based"])
        prediction, truth, baseline, seed_mask, valid, coverage = load_patch(
            source, output, patch, first_target_day, patch_count)
    points = choose_points(seed_mask)
    selected, point_rows, scale_info = derive_maps_and_points(
        prediction, truth, baseline, valid, points, days,
        args.denominator_floor, relative_error_cap=args.relative_error_cap)
    output_dir = (args.output_dir.resolve() if args.output_dir is not None else
                  rollout_dir / f"figures_high_ocean_patch_{patch:03d}")
    output_dir.mkdir(parents=True, exist_ok=False)
    point_fields = ("lead_day", "point_id", "row", "col", "valid", "truth_speed",
                    "model_relative_error", "baseline_relative_error",
                    "denominator_floored")
    with (output_dir / "point_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields)
        writer.writeheader()
        writer.writerows(point_rows)
    metadata = {
        "source_h5": str(source_path),
        "predictions_h5": str(prediction_path),
        "patch_index": patch,
        "ocean_fraction_from_seed_mask": ocean_fraction,
        "selected_points": points,
        "selected_lead_days": list(days),
        "first_target_day_index_zero_based": first_target_day,
        "relative_error_formula": "sqrt((pred_u-GT_u)^2+(pred_v-GT_v)^2) / max(sqrt(GT_u^2+GT_v^2), denominator_floor)",
        "persistence_definition": "U/V from last observed day, repeated for all future days",
        "map_mask": "target valid ocean intersected with last-observed valid ocean",
        "target_mask_coverage": coverage,
        "scale": scale_info,
        "velocity_units": "not recorded in source H5; plots use native H5 units",
        "figures": ["speed_comparison", "relative_error_comparison",
                    "four_point_error_36day"],
        "status": "draft_for_review",
    }
    with (output_dir / "plot_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
    plt = set_style()
    make_map_figure(plt, selected, patch, ocean_fraction, points, scale_info,
                    "speed", output_dir)
    make_map_figure(plt, selected, patch, ocean_fraction, points, scale_info,
                    "relative_error", output_dir)
    make_point_figure(plt, point_rows, points, patch, ocean_fraction, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "patch": patch,
                      "ocean_fraction": ocean_fraction, "points": points,
                      "speed_colorbar_max": scale_info["speed_colorbar_max"],
                      "relative_error_colorbar_max": args.relative_error_cap},
                     indent=2), flush=True)


if __name__ == "__main__":
    main(parse_args())
