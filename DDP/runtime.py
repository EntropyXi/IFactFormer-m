"""Thin torch.distributed lifecycle and collective helpers."""

import os

import torch
import torch.distributed as dist


class DistributedRuntime:
    def __init__(self, requested_device):
        required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
        if any(name not in os.environ for name in required):
            raise RuntimeError("DDP requires torchrun (missing distributed environment variables)")
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        if self.world_size < 2:
            raise ValueError("DDP requires at least two processes")
        if torch.device(requested_device).type == "cuda":
            if self.local_rank >= torch.cuda.device_count():
                raise RuntimeError("LOCAL_RANK exceeds visible CUDA devices")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            backend = "nccl"
        elif torch.device(requested_device).type == "cpu":
            self.device = torch.device("cpu")
            backend = "gloo"
        else:
            raise ValueError("DDP supports CUDA training or CPU/Gloo testing")
        # Some Windows CPU wheels lack libuv, and their torchrun launcher cannot
        # initialize its own TCPStore. Direct rank launch plus this URL supports
        # local Gloo tests; normal Linux/CUDA training continues to use env://.
        if os.name == "nt" and self.device.type == "cpu":
            init_method = (f"tcp://{os.environ['MASTER_ADDR']}:"
                           f"{os.environ['MASTER_PORT']}?use_libuv=0")
            dist.init_process_group(backend=backend, init_method=init_method,
                                    rank=self.rank, world_size=self.world_size)
        else:
            dist.init_process_group(backend=backend, init_method="env://")

    @property
    def primary(self):
        return self.rank == 0

    def broadcast_object(self, value):
        values = [value if self.primary else None]
        dist.broadcast_object_list(values, src=0, device=self.device)
        return values[0]

    def broadcast_stats(self, mean, std):
        values = torch.empty(4, dtype=torch.float32, device=self.device)
        if self.primary:
            values[:2] = mean.to(self.device)
            values[2:] = std.to(self.device)
        dist.broadcast(values, src=0)
        return values[:2].clone(), values[2:].clone()

    def any_true(self, local_value):
        flag = torch.tensor(int(bool(local_value)), dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def mean(self, local_value):
        value = torch.tensor(float(local_value), dtype=torch.float64, device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return (value / self.world_size).item()

    def sum_int(self, local_value):
        value = torch.tensor(int(local_value), dtype=torch.int64, device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return int(value.item())

    def weighted_mean(self, local_sum, local_count):
        values = torch.tensor((local_sum, local_count), dtype=torch.float64, device=self.device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        if values[1].item() == 0:
            raise ValueError("No validation/test samples were evaluated")
        return (values[0] / values[1]).item()

    def gather_rng_states(self):
        state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }
        states = [None] * self.world_size
        dist.all_gather_object(states, state)
        return states

    def restore_rng_state(self, state):
        torch.set_rng_state(state["torch"])
        if self.device.type == "cuda" and state["cuda"] is not None:
            torch.cuda.set_rng_state(state["cuda"], self.device)

    def barrier(self):
        dist.barrier()

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
