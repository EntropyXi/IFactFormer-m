"""Four-rank CPU/Gloo smoke test for DDP checkpoint and uneven shards.

Run from the repository root with ``python -B tests/test_ddp_smoke.py``.
"""

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


def make_fixture(directory):
    h5_path = directory / "tiny.h5"
    config_path = directory / "tiny.yml"
    days, patches, height, width = 16, 2, 4, 5
    rows = days * patches
    rng = np.random.default_rng(42)
    data = rng.normal(size=(rows, 2, height, width)).astype(np.float32)
    mask = np.zeros((rows, height, width), dtype=np.bool_)
    mask[..., 0] = True
    data *= 0.2
    data[:, 0] += np.arange(rows, dtype=np.float32)[:, None, None] * 0.01
    data[:, 1] += np.arange(rows, dtype=np.float32)[:, None, None] * 0.02
    data[:, :, :, 0] = np.nan
    lat = np.broadcast_to(np.linspace(-20, 20, height, dtype=np.float32)[None, :, None],
                          (rows, height, width)).copy()
    lon = np.broadcast_to(np.linspace(170, 190, width, dtype=np.float32)[None, None, :],
                          (rows, height, width)).copy()
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("uovo_data", data=data)
        h5_file.create_dataset("mask", data=mask)
        h5_file.create_dataset("lat", data=lat)
        h5_file.create_dataset("lon", data=lon)
    config = {
        "log_dir": str(directory),
        "model": {"in_dim": 3, "out_dim": 2, "in_time_window": 7,
                  "dim": 24, "heads": 2, "depth": 1, "dim_head": 16,
                  "n_layer": 1, "model": "IFactFormer_m"},
        "training": {"epochs": 2, "lr": 0.0005, "batch_size": 1,
                     "scheduler_step": 5, "scheduler_gamma": 0.7},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return h5_path, config_path


def invoke(h5_path, config_path, run_dir=None, resume=None, stop_after=None):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [sys.executable, "-B", "main.py", "--ddp", "--device", "cpu",
               "--config", str(config_path), "--h5", str(h5_path),
               "--patches-per-day", "2", "--num-workers", "0",
               "--checkpoint-every-steps", "2", "--log-every-steps", "2"]
    if run_dir is not None:
        command += ["--run-dir", str(run_dir)]
    if resume is not None:
        command += ["--resume", str(resume)]
    if stop_after is not None:
        command += ["--stop-after-steps", str(stop_after)]
    processes = []
    for rank in range(4):
        environment = os.environ.copy()
        environment.update({"RANK": str(rank), "LOCAL_RANK": str(rank),
                            "WORLD_SIZE": "4", "MASTER_ADDR": "127.0.0.1",
                            "MASTER_PORT": str(port), "OMP_NUM_THREADS": "1"})
        processes.append(subprocess.Popen(command, cwd=ROOT, env=environment,
                                          text=True, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE))
    try:
        outputs = [process.communicate(timeout=180) for process in processes]
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
        raise
    for rank, (process, (stdout, stderr)) in enumerate(zip(processes, outputs)):
        if process.returncode:
            raise AssertionError(f"Rank {rank} failed ({process.returncode}):\n{stdout}\n{stderr}")
        if rank == 0:
            print(stdout[-1200:])


def invoke_single(h5_path, config_path, run_dir):
    command = [sys.executable, "-B", "main.py", "--device", "cpu",
               "--config", str(config_path), "--h5", str(h5_path),
               "--patches-per-day", "2", "--num-workers", "0",
               "--run-dir", str(run_dir)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                            timeout=180)
    if result.returncode:
        raise AssertionError(f"Single-process regression failed:\n{result.stdout}\n{result.stderr}")


def main():
    with tempfile.TemporaryDirectory(prefix="ifactformer_ddp_smoke_") as temporary:
        directory = Path(temporary)
        h5_path, config_path = make_fixture(directory)
        resumed = directory / "resumed"
        full = directory / "full"
        invoke(h5_path, config_path, run_dir=resumed, stop_after=2)
        partial = torch.load(resumed / "checkpoint_latest.pt", weights_only=True)
        assert partial["format_version"] == 2
        assert partial["world_size"] == 4
        assert partial["local_offset"] == 2
        assert partial["global_step"] == 2
        assert partial["samples_per_rank"] == 4  # 14 true + 2 padded
        invoke(h5_path, config_path, resume=resumed)
        invoke(h5_path, config_path, run_dir=full)
        resumed_state = torch.load(resumed / "checkpoint_latest.pt", weights_only=True)
        full_state = torch.load(full / "checkpoint_latest.pt", weights_only=True)
        assert resumed_state["global_step"] == full_state["global_step"] == 8
        assert resumed_state["epoch"] == full_state["epoch"] == 2
        assert resumed_state["scheduler"] == full_state["scheduler"]
        assert resumed_state["best_val"] == full_state["best_val"]
        assert resumed_state["test_loss"] == full_state["test_loss"]
        max_difference = max((resumed_state["model"][name] - full_state["model"][name]).abs().max().item()
                             for name in resumed_state["model"])
        print("maximum resumed/full parameter difference:", max_difference)
        assert max_difference < 1e-6
        single = directory / "single"
        invoke_single(h5_path, config_path, single)
        single_state = torch.load(single / "checkpoint_latest.pt", weights_only=True)
        assert single_state["format_version"] == 1 and single_state["epoch"] == 2
        print("PASS: four ranks, uneven split, numerically consistent resume, single-GPU regression")


if __name__ == "__main__":
    main()
