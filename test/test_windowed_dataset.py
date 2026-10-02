"""CPU checks for non-overlapping spatial windows and stitched evaluation."""

import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from libs.copernicus_h5_dataset import CopernicusH5Dataset, compute_train_uv_stats
from libs.model import Model
from libs.training_common import split_counts
from libs.utils import MaskedLpLoss, dict2namespace


def make_fixture(root):
    days, patches, height, width = 40, 2, 4, 4
    rows = days * patches
    rng = np.random.default_rng(2026)
    uv = rng.normal(0, 0.02, (rows, 2, height, width)).astype(np.float32)
    land = np.zeros((rows, height, width), dtype=np.bool_)
    land[:, :2, :2] = True  # One completely land-covered 2x2 window.
    uv[:, :, :2, :2] = np.nan
    lat = np.broadcast_to(np.arange(height, dtype=np.float32)[None, :, None],
                          (rows, height, width)).copy()
    lon = np.broadcast_to(np.arange(width, dtype=np.float32)[None, None, :],
                          (rows, height, width)).copy()
    path = root / "windowed.h5"
    with h5py.File(path, "w") as h5:
        for key, value in (("uovo_data", uv), ("mask", land), ("lat", lat), ("lon", lon)):
            h5.create_dataset(key, data=value)
    return path


def test_exact_112_layout(root):
    path = root / "sparse_448.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("uovo_data", shape=(16, 2, 448, 448), dtype="f4",
                          chunks=(1, 2, 112, 112), fillvalue=0)
        for key, dtype in (("mask", "bool"), ("lat", "f4"), ("lon", "f4")):
            h5.create_dataset(key, shape=(16, 448, 448), dtype=dtype,
                              chunks=(1, 112, 112), fillvalue=0)
    dataset = CopernicusH5Dataset(path, patches_per_day=2, input_days=7,
                                 window_size=112)
    try:
        assert dataset.source_spatial_shape == (448, 448)
        assert dataset.spatial_shape == (112, 112)
        assert dataset.tiles_per_patch == 16
        assert dataset.samples_per_day == 32
        assert len(dataset) == 32
        assert dataset.decode_index(31) == (0, 1, 3, 3)
        assert dataset.window_bounds(3, 3) == (336, 448, 336, 448)
        x, y, valid, positions = dataset[31]
        assert x.shape == (7, 112, 112, 3)
        assert y.shape == (112, 112, 2)
        assert valid.shape == (112, 112)
        assert positions["absolute"].shape == (112, 112, 3)
    finally:
        dataset.close()


def test_dataset_and_loss(path):
    full = CopernicusH5Dataset(path, patches_per_day=2, input_days=7)
    tiled = CopernicusH5Dataset(path, patches_per_day=2, input_days=7, window_size=2)
    try:
        assert tiled.tiles_per_patch == 4
        assert tiled.samples_per_day == 8
        assert len(tiled) == len(full) * 4
        assert split_counts(tiled) == tuple(count * 4 for count in split_counts(full))
        assert tiled.decode_index(7) == (0, 1, 1, 1)
        assert tiled.decode_index(8) == (1, 0, 0, 0)

        full_x, full_y, _, _ = full[0]
        for tile in range(4):
            x, y, valid, pos = tiled[tile]
            row, col = divmod(tile, 2)
            ys, xs = slice(row * 2, row * 2 + 2), slice(col * 2, col * 2 + 2)
            assert x.shape == (7, 2, 2, 3) and y.shape == (2, 2, 2)
            assert valid.shape == (2, 2)
            assert pos["absolute"].shape == (2, 2, 3)
            torch.testing.assert_close(x, full_x[:, ys, xs])
            torch.testing.assert_close(y, full_y[ys, xs])
        assert not tiled[0][2].any()
        full_stats = compute_train_uv_stats(full, split_counts(full)[0], chunk_rows=3)
        tile_stats = compute_train_uv_stats(tiled, split_counts(tiled)[0], chunk_rows=3)
        for original, windowed in zip(full_stats, tile_stats):
            torch.testing.assert_close(original, windowed)

        prediction = torch.ones((2, 2, 2, 2), requires_grad=True)
        target = torch.ones_like(prediction)
        mask = torch.zeros((2, 2, 2), dtype=torch.bool)
        mask[1] = True
        loss = MaskedLpLoss()(prediction, target, mask)
        assert loss.item() == 0.0
        loss.backward()
        assert torch.all(prediction.grad == 0)

        model_config = {"in_dim": 3, "out_dim": 2, "in_time_window": 7,
                        "dim": 24, "heads": 2, "depth": 1,
                        "dim_head": 16, "n_layer": 1, "model": "IFactFormer_m"}
        model = Model(dict2namespace(model_config))
        x0, y0, v0, p0 = tiled[0]
        x1, y1, v1, p1 = tiled[1]
        positions = {key: torch.stack((p0[key], p1[key])) for key in p0}
        output = model(torch.stack((x0, x1)), positions)
        assert output.shape == (2, 2, 2, 2)
        targets = torch.stack((y0, y1))
        valid = torch.stack((v0, v1))
        losses = MaskedLpLoss(reduction=False)(output, targets, valid)
        training_loss = losses.sum() / valid.flatten(1).any(dim=1).sum().clamp_min(1)
        assert torch.isfinite(training_loss)
        training_loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters())
    finally:
        tiled.close()
        full.close()


def test_windowed_evaluators(path, root):
    dataset = CopernicusH5Dataset(path, patches_per_day=2, input_days=7, window_size=2)
    run_dir = root / "run"
    run_dir.mkdir()
    config = {"in_dim": 3, "out_dim": 2, "in_time_window": 7, "dim": 24,
              "heads": 2, "depth": 1, "dim_head": 16, "n_layer": 1,
              "model": "IFactFormer_m"}
    model = Model(dict2namespace(config))
    torch.save({"model_config": config, "h5_path": str(path), "patches_per_day": 2,
                "window_size": 2, "splits": split_counts(dataset),
                "uv_mean": torch.zeros(2), "uv_std": torch.ones(2)},
               run_dir / "checkpoint_latest.pt")
    torch.save({"model": model.state_dict(), "epoch": 0}, run_dir / "checkpoint_best.pth")
    dataset.close()

    one_step = run_dir / "one_step"
    command = [sys.executable, "-B", str(TEST_DIR / "evaluate_one_step.py"),
               "--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0",
               "--plot-count", "0", "--max-samples", "8", "--output-dir", str(one_step)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(f"One-step evaluation failed:\n{result.stdout}\n{result.stderr}")
    summary = json.loads((one_step / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluated_test_samples"] == 8
    assert summary["valid_ocean_samples"] == 6
    with (one_step / "per_sample.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(row["patch_index"]), int(row["tile_row"]), int(row["tile_col"]))
            for row in rows] == [(patch, row, col)
                                for patch in range(2) for row in range(2) for col in range(2)]
    assert rows[0]["model_relative_l2"] == ""

    rollout = run_dir / "rollout"
    command = [sys.executable, "-B", str(TEST_DIR / "rollout_36day.py"),
               "--run-dir", str(run_dir), "--device", "cpu", "--horizon-days", "2",
               "--max-patches", "1", "--output-dir", str(rollout)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(f"Windowed rollout failed:\n{result.stdout}\n{result.stderr}")
    summary = json.loads((rollout / "summary.json").read_text(encoding="utf-8"))
    assert summary["windows_per_patch"] == 4
    with h5py.File(rollout / "predictions.h5", "r") as output:
        assert output["pred_uv"].shape == (2, 1, 2, 4, 4)
        assert np.isfinite(output["pred_uv"][:]).all()
        assert np.all(output["pred_uv"][:, :, :, :2, :2] == 0)


def test_training_entrypoint(path, root):
    config = {"log_dir": str(root),
              "model": {"in_dim": 3, "out_dim": 2, "in_time_window": 7,
                        "dim": 24, "heads": 2, "depth": 1,
                        "dim_head": 16, "n_layer": 1, "model": "IFactFormer_m"},
              "training": {"epochs": 1, "lr": 0.0005, "batch_size": 2,
                           "scheduler_step": 5, "scheduler_gamma": 0.7}}
    config_path = root / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = root / "train"
    command = [sys.executable, "-B", str(ROOT / "main.py"), "--config", str(config_path),
               "--h5", str(path), "--patches-per-day", "2", "--window-size", "2",
               "--device", "cpu", "--num-workers", "0", "--stop-after-steps", "1",
               "--checkpoint-every-steps", "1", "--run-dir", str(run_dir)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(f"Windowed training failed:\n{result.stdout}\n{result.stderr}")
    checkpoint = torch.load(run_dir / "checkpoint_latest.pt", map_location="cpu",
                            weights_only=True)
    assert checkpoint["window_size"] == 2
    assert tuple(checkpoint["splits"]) == (208, 24, 32)
    assert checkpoint["global_step"] == 1


def test_ddp_training_entrypoint(path, root):
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    run_dir = root / "ddp_train"
    command = [sys.executable, "-B", str(ROOT / "main.py"), "--ddp",
               "--config", str(root / "config.yml"), "--h5", str(path),
               "--patches-per-day", "2", "--window-size", "2", "--device", "cpu",
               "--num-workers", "0", "--stop-after-steps", "1",
               "--checkpoint-every-steps", "1", "--run-dir", str(run_dir)]
    processes = []
    try:
        for rank in range(2):
            env = os.environ.copy()
            env.update({"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2",
                        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, text=True,
                                              errors="replace"))
        outputs = [process.communicate(timeout=120)[0] for process in processes]
        if any(process.returncode for process in processes):
            raise AssertionError(f"Windowed DDP training failed:\n{outputs}")
        checkpoint = torch.load(run_dir / "checkpoint_latest.pt", map_location="cpu",
                                weights_only=True)
        assert checkpoint["window_size"] == 2
        assert checkpoint["world_size"] == 2
        assert checkpoint["global_step"] == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="ifact_windowed_test_") as temporary:
        root = Path(temporary)
        test_exact_112_layout(root)
        path = make_fixture(root)
        test_dataset_and_loss(path)
        test_windowed_evaluators(path, root)
        test_training_entrypoint(path, root)
        if os.environ.get("IFACT_TEST_DDP") == "1":
            test_ddp_training_entrypoint(path, root)
    print("PASS: tiling, date split, train stats, all-land loss, evaluators, and training")
