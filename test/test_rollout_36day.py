"""CPU smoke and leakage checks for the free-rollout evaluator.

Run from the repository root: python -B test/test_rollout_36day.py
"""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch


TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parent
for directory in (str(ROOT), str(TEST_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from rollout_36day import iter_free_rollout, load_seed
from evaluate_one_step import load_run
from libs.copernicus_h5_dataset import CopernicusH5Dataset
from libs.model import Model
from libs.training_common import split_counts
from libs.utils import dict2namespace, normalize_uv_input


class RecordingPlusOne(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_inputs = []

    def forward(self, x, _positions):
        self.last_inputs.append(x[:, -1, ..., :2].clone())
        return x[:, -1, ..., :2] + 1.0


def test_recurrence_uses_predictions():
    history = torch.zeros((1, 7, 2, 3, 3), dtype=torch.float32)
    history[..., 2] = 1.0
    history[:, -1, ..., :2] = 3.0
    history[:, :, 0, 0, 2] = 0.0
    model = RecordingPlusOne()
    predictions = list(iter_free_rollout(model, history, None, torch.zeros(2),
                                         torch.ones(2), horizon_days=3))
    for lead, prediction in enumerate(predictions, start=1):
        assert torch.all(prediction[0, 1:, 1:, :] == 3.0 + lead)
        assert torch.all(prediction[0, 0, 0, :] == 0.0)
    assert torch.all(model.last_inputs[0][0, 1:, 1:, :] == 3.0)
    assert torch.all(model.last_inputs[1][0, 1:, 1:, :] == 4.0)
    assert torch.all(model.last_inputs[2][0, 1:, 1:, :] == 5.0)


def make_fixture(root):
    days, patches, height, width = 40, 2, 4, 5
    rows = days * patches
    rng = np.random.default_rng(123)
    uv = rng.normal(0, 0.02, size=(rows, 2, height, width)).astype(np.float32)
    uv += np.arange(rows, dtype=np.float32)[:, None, None, None] * 0.001
    land = np.zeros((rows, height, width), dtype=np.bool_)
    land[..., 0] = True
    uv[:, :, :, 0] = np.nan
    lat = np.broadcast_to(np.linspace(-20, 20, height, dtype=np.float32)[None, :, None],
                          (rows, height, width)).copy()
    lon = np.broadcast_to(np.linspace(170, 190, width, dtype=np.float32)[None, None, :],
                          (rows, height, width)).copy()
    path = root / "tiny.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("uovo_data", data=uv)
        h5.create_dataset("mask", data=land)
        h5.create_dataset("lat", data=lat)
        h5.create_dataset("lon", data=lon)
    config = {"in_dim": 3, "out_dim": 2, "in_time_window": 7, "dim": 24,
              "heads": 2, "depth": 1, "dim_head": 16, "n_layer": 1,
              "model": "IFactFormer_m"}
    dataset = CopernicusH5Dataset(path, patches_per_day=patches, input_days=7)
    splits = split_counts(dataset)
    dataset.close()
    run_dir = root / "run"
    run_dir.mkdir()
    model = Model(dict2namespace(config))
    torch.save({"model_config": config, "h5_path": str(path), "patches_per_day": patches,
                "splits": splits, "uv_mean": torch.zeros(2), "uv_std": torch.ones(2)},
               run_dir / "checkpoint_latest.pt")
    torch.save({"model": model.state_dict(), "epoch": 0},
               run_dir / "checkpoint_best.pth")
    return path, run_dir, splits


def test_cli_output():
    with tempfile.TemporaryDirectory(prefix="ifact_rollout_smoke_") as temporary:
        root = Path(temporary)
        h5_path, run_dir, splits = make_fixture(root)
        output_dir = run_dir / "rollout_smoke"
        command = [sys.executable, "-B", str(TEST_DIR / "rollout_36day.py"),
                   "--run-dir", str(run_dir), "--device", "cpu",
                   "--horizon-days", "3", "--max-patches", "1",
                   "--output-dir", str(output_dir)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                timeout=120)
        if result.returncode:
            raise AssertionError(f"Rollout failed:\n{result.stdout}\n{result.stderr}")
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["horizon_days"] == 3
        assert summary["evaluated_patches"] == 1
        assert not summary["complete_test_horizon"]
        assert summary["first_target_day_index"] == (splits[0] + splits[1]) // 2 + 7
        with (output_dir / "per_patch_day.csv").open(newline="", encoding="utf-8") as handle:
            detail_rows = list(csv.DictReader(handle))
        with (output_dir / "per_lead.csv").open(newline="", encoding="utf-8") as handle:
            lead_rows = list(csv.DictReader(handle))
        assert len(detail_rows) == len(lead_rows) == 3
        with h5py.File(output_dir / "predictions.h5", "r") as predictions:
            assert predictions.attrs["status"] == "complete"
            assert predictions["pred_uv"].shape == (3, 1, 2, 4, 5)
            assert np.isfinite(predictions["pred_uv"][:]).all()
            assert np.all(predictions["pred_uv"][:, :, :, :, 0] == 0)
            actual_first = predictions["pred_uv"][0, 0]
        latest, best, model, mean, std, _, _ = load_run(run_dir, torch.device("cpu"))
        with h5py.File(h5_path, "r") as source, torch.inference_mode():
            history, positions, seed_valid = load_seed(
                source, summary["first_input_day_index"], 0, 2, 7,
                torch.device("cpu"))
            expected_first = model(normalize_uv_input(history, mean, std), positions) * std + mean
            expected_first = torch.where(torch.from_numpy(seed_valid)[None, ..., None],
                                         expected_first, 0.0)
        np.testing.assert_allclose(actual_first, np.moveaxis(expected_first[0].numpy(), -1, 0),
                                   rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_recurrence_uses_predictions()
    test_cli_output()
    print("PASS: prediction-only recurrence, partial CLI, H5/CSV summary, and lead-1 match")
