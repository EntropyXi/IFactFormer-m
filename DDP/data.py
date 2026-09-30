"""Deterministic, resumable per-rank indexing for a map-style H5 dataset."""

import math

import torch


def samples_per_rank(sample_count, world_size):
    return math.ceil(sample_count / world_size)


def train_indices(sample_count, world_size, rank, seed, epoch, offset=0):
    """Shard a global permutation; pad its tail so all ranks take equal steps.

    Padding repeats at most ``world_size - 1`` training samples per epoch.
    ``offset`` counts *local* samples already processed, enabling mid-epoch resume.
    """
    per_rank = samples_per_rank(sample_count, world_size)
    if not 0 <= offset <= per_rank:
        raise ValueError("Invalid per-rank training sample offset")
    order = torch.randperm(sample_count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    padded_count = per_rank * world_size
    padding = padded_count - sample_count
    if padding:
        order += (order * math.ceil(padding / sample_count))[:padding]
    return order[rank:padded_count:world_size][offset:]


def evaluation_indices(start, count, world_size, rank):
    """Shard validation/test samples without padding or duplication."""
    return range(start + rank, start + count, world_size)
