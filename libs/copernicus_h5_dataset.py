"""Read seven-day UV windows and geographic metadata on demand."""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def build_patch_positions(lat, lon):
    """Build per-patch rotary axes and periodic absolute coordinates."""
    lat_axis = np.deg2rad(lat[:, 0]).astype(np.float32)
    lon_axis = np.deg2rad(lon[0, :]).astype(np.float64)
    lon_unwrapped = np.unwrap(lon_axis).astype(np.float32)
    lon_axis = lon_axis.astype(np.float32)
    height, width = lat.shape
    absolute = np.stack((
        np.broadcast_to(lat_axis[:, None], (height, width)),
        np.broadcast_to(np.sin(lon_axis)[None, :], (height, width)),
        np.broadcast_to(np.cos(lon_axis)[None, :], (height, width)),
    ), axis=-1).astype(np.float32)
    return {
        "axis_lat": torch.from_numpy(lat_axis[:, None].copy()),
        "axis_lon": torch.from_numpy(lon_unwrapped[:, None].copy()),
        "absolute": torch.from_numpy(absolute),
    }


def compute_train_uv_stats(dataset, train_sample_count, chunk_rows=16):
    """Stream ocean-only U/V statistics over the training input windows."""
    if not 0 < train_sample_count <= len(dataset):
        raise ValueError("train_sample_count must be between 1 and len(dataset)")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    if train_sample_count % dataset.tiles_per_patch:
        raise ValueError("training sample count must include complete spatial tilings")
    # Each complete tiling covers exactly the pixels of one original H5 patch.
    train_patch_count = train_sample_count // dataset.tiles_per_patch
    patches = dataset.patches_per_day
    window = dataset.input_days
    full_days, partial_patches = divmod(train_patch_count, patches)
    starts_per_patch = full_days + (np.arange(patches) < partial_patches)
    last_start_day = full_days if partial_patches else full_days - 1
    row_limit = (last_start_day + window) * patches

    sums = np.zeros(2, dtype=np.float64)
    squared_sums = np.zeros(2, dtype=np.float64)
    valid_count = 0.0
    with h5py.File(dataset.path, "r") as h5_file:
        data = h5_file["uovo_data"]
        land_mask = h5_file["mask"]
        for start in range(0, row_limit, chunk_rows):
            end = min(start + chunk_rows, row_limit)
            rows = np.arange(start, end)
            day, patch = divmod(rows, patches)
            first_start = np.maximum(0, day - window + 1)
            last_start = np.minimum(day, starts_per_patch[patch] - 1)
            weights = np.maximum(0, last_start - first_start + 1).astype(np.float64)
            if not weights.any():
                continue

            uv = data[start:end]
            valid = (~land_mask[start:end]) & np.isfinite(uv).all(axis=1)
            valid_count += np.sum(valid * weights[:, None, None], dtype=np.float64)
            for channel in range(2):
                values = np.where(valid, uv[:, channel], 0).astype(np.float64)
                weighted = values * weights[:, None, None]
                sums[channel] += np.sum(weighted, dtype=np.float64)
                squared_sums[channel] += np.sum(weighted * values, dtype=np.float64)

    if valid_count <= 1:
        raise ValueError("training windows contain too few valid ocean pixels")
    mean = sums / valid_count
    variance = (squared_sums - sums * sums / valid_count) / (valid_count - 1)
    std = np.sqrt(np.maximum(variance, 0.0))
    if np.any(std < 1e-8):
        raise ValueError("training U/V standard deviation is zero")
    return (torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32))


class CopernicusH5Dataset(Dataset):
    """Return velocity + ocean mask, target, target mask and coordinates."""

    def __init__(self, path, patches_per_day=131, input_days=7, window_size=None):
        self.path = os.fspath(path)
        self.patches_per_day = patches_per_day
        self.input_days = input_days
        if patches_per_day <= 0 or input_days <= 0:
            raise ValueError("patches_per_day and input_days must be positive")

        # Read only metadata here; worker processes open their own H5 handle.
        with h5py.File(self.path, "r") as h5_file:
            shape = h5_file["uovo_data"].shape
        if len(shape) != 4 or shape[1] != 2:
            raise ValueError(f"Expected uovo_data [N, 2, H, W], got {shape}")
        if shape[0] % patches_per_day:
            raise ValueError("uovo_data length is not divisible by patches_per_day")

        self.num_days = shape[0] // patches_per_day
        self.source_spatial_shape = shape[2:]
        if window_size is not None and (window_size <= 0 or
                                        any(side % window_size for side in shape[2:])):
            raise ValueError(f"window_size {window_size} must divide H5 spatial shape {shape[2:]}")
        self.window_size = window_size
        self.window_height = window_size or shape[2]
        self.window_width = window_size or shape[3]
        self.window_rows = shape[2] // self.window_height
        self.window_cols = shape[3] // self.window_width
        self.tiles_per_patch = self.window_rows * self.window_cols
        self.samples_per_day = patches_per_day * self.tiles_per_patch
        self.spatial_shape = (self.window_height, self.window_width)
        self._h5_file = None
        self._data = None
        self._mask = None
        self._lat = None
        self._lon = None
        self._pid = None

    def __len__(self):
        return max(0, self.num_days - self.input_days) * self.samples_per_day

    def decode_index(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        day, sample_in_day = divmod(index, self.samples_per_day)
        patch, tile = divmod(sample_in_day, self.tiles_per_patch)
        tile_row, tile_col = divmod(tile, self.window_cols)
        return day, patch, tile_row, tile_col

    def window_bounds(self, tile_row, tile_col):
        if not 0 <= tile_row < self.window_rows or not 0 <= tile_col < self.window_cols:
            raise IndexError((tile_row, tile_col))
        y0 = tile_row * self.window_height
        x0 = tile_col * self.window_width
        return y0, y0 + self.window_height, x0, x0 + self.window_width

    def _get_data(self):
        pid = os.getpid()
        if self._h5_file is None or self._pid != pid:
            self._h5_file = h5py.File(self.path, "r")
            self._data = self._h5_file["uovo_data"]
            self._mask = self._h5_file["mask"]
            self._lat = self._h5_file["lat"]
            self._lon = self._h5_file["lon"]
            self._pid = pid
        return self._data

    def __getitem__(self, index):
        day, patch, tile_row, tile_col = self.decode_index(index)
        y0, y1, x0, x1 = self.window_bounds(tile_row, tile_col)
        first_row = day * self.patches_per_day + patch
        target_row = (day + self.input_days) * self.patches_per_day + patch
        data = self._get_data()
        # Rows for the same patch on consecutive days are 131 apart.
        x = data[first_row:target_row:self.patches_per_day, :, y0:y1, x0:x1]
        y = data[target_row, :, y0:y1, x0:x1]
        # In this H5, mask=True marks invalid/land cells (where velocity is NaN).
        land_masks = self._mask[first_row:target_row:self.patches_per_day, y0:y1, x0:x1]
        target_land_mask = self._mask[target_row, y0:y1, x0:x1]
        input_valid = (~land_masks) & np.isfinite(x).all(axis=1)
        target_valid = (~target_land_mask) & np.isfinite(y).all(axis=0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        x *= input_valid[:, None, ...]
        y *= target_valid[None, ...]
        x = np.concatenate((np.moveaxis(x, 1, -1), input_valid[..., None].astype(np.float32)), axis=-1)
        positions = build_patch_positions(self._lat[first_row, y0:y1, x0:x1],
                                          self._lon[first_row, y0:y1, x0:x1])
        return (torch.from_numpy(x),
                torch.from_numpy(np.moveaxis(y, 0, -1)),
                torch.from_numpy(target_valid),
                positions)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_data"] = None
        state["_mask"] = None
        state["_lat"] = None
        state["_lon"] = None
        state["_pid"] = None
        return state

    def close(self):
        if getattr(self, "_h5_file", None) is not None:
            self._h5_file.close()
            self._h5_file = None
            self._data = None
            self._mask = None
            self._lat = None
            self._lon = None
            self._pid = None

    def __del__(self):
        self.close()
