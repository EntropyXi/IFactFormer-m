"""A repeated shared attention block must keep its gradients under checkpointing."""

from types import SimpleNamespace

import numpy as np
import torch

from libs.copernicus_h5_dataset import build_patch_positions
from libs.model import Model


def test_checkpointed_repeated_block_matches_uncheckpointed_gradients():
    torch.manual_seed(17)
    config = SimpleNamespace(in_dim=3, out_dim=2, in_time_window=7,
                             dim=24, heads=2, depth=1, dim_head=16,
                             n_layer=3, model="IFactFormer_m")
    reference = Model(config).train()
    checkpointed = Model(config, activation_checkpoint=True).train()
    checkpointed.load_state_dict(reference.state_dict())

    x = torch.randn(1, 7, 4, 5, 3)
    x[..., 2] = 1.0
    x[:, :, 0, 0, 2] = 0.0
    lat = np.broadcast_to(np.linspace(-20, 20, 4, dtype=np.float32)[:, None],
                          (4, 5))
    lon = np.broadcast_to(np.linspace(170, 190, 5, dtype=np.float32)[None, :],
                          (4, 5))
    positions = {name: value.unsqueeze(0)
                 for name, value in build_patch_positions(lat, lon).items()}

    reference_output = reference(x, positions)
    checkpointed_output = checkpointed(x, positions)
    torch.testing.assert_close(checkpointed_output, reference_output)
    reference_output.square().mean().backward()
    checkpointed_output.square().mean().backward()
    for (name, parameter), (other_name, other_parameter) in zip(
            reference.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        if parameter.grad is None:
            assert other_parameter.grad is None
        else:
            torch.testing.assert_close(other_parameter.grad, parameter.grad,
                                       rtol=1e-5, atol=1e-6)
