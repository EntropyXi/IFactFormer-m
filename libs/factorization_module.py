import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Union, Tuple, List, Optional
from libs.positional_encoding_module import RotaryEmbedding, apply_rotary_pos_emb, SirenNet
from libs.basics import PreNorm, PostNorm, GeAct, MLP, masked_instance_norm
from libs.attention import LowRankKernel

class PoolingReducer(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super().__init__()
        self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_ffn = PreNorm(in_dim, MLP([hidden_dim, hidden_dim, out_dim], GeAct(nn.GELU())))

    def forward(self, x, valid_mask=None):
        # note that the dimension to be pooled will be the last dimension
        # x: b nx ... c
        x = self.to_in(x)
        # pool all spatial dimension but the first one
        ndim = len(x.shape)
        if valid_mask is None:
            x = x.mean(dim=tuple(range(2, ndim-1)))
        else:
            if x.ndim != 4 or valid_mask.shape != x.shape[:-1]:
                raise ValueError("masked pooling expects x [B, H, W, C] and mask [B, H, W]")
            weights = valid_mask.unsqueeze(-1).to(dtype=x.dtype)
            counts = weights.sum(dim=2).clamp_min(1)
            x = (x * weights).sum(dim=2) / counts
        x = self.out_ffn(x)
        if valid_mask is not None:
            x = x * valid_mask.any(dim=2).unsqueeze(-1)
        return x  # b nx c


class MaskedInstanceNorm2d(nn.Module):
    """Instance normalization using only valid spatial locations."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, valid_mask):
        if x.ndim != 4 or valid_mask.shape != (x.shape[0], *x.shape[2:]):
            raise ValueError("masked instance norm expects x [B, C, H, W] and mask [B, H, W]")
        acc = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        weights = valid_mask.unsqueeze(1).to(dtype=acc.dtype)
        counts = weights.sum(dim=(2, 3), keepdim=True).clamp_min(1)
        mean = (acc * weights).sum(dim=(2, 3), keepdim=True) / counts
        variance = ((acc - mean).square() * weights).sum(dim=(2, 3), keepdim=True) / counts
        return (((acc - mean) * torch.rsqrt(variance + self.eps)) * weights).to(dtype=x.dtype)

class FABlock2D_m(nn.Module):
    # contains factorization and attention on each axis
    def __init__(self,
                 dim,
                 dim_head,
                 latent_dim,
                 heads,
                 dim_out,
                 use_rope=True,
                 kernel_multiplier=3,
                 scaling_factor=1.0):
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.dim_head = dim_head
        self.in_norm = nn.LayerNorm(dim)
        self.to_v = nn.Linear(self.dim, heads * dim_head, bias=False)
        self.to_in = nn.Linear(self.dim, self.dim, bias=False)

        self.to_x = nn.Sequential(
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )
        self.to_y = nn.Sequential(
            Rearrange('b nx ny c -> b ny nx c'),
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )

        positional_encoding = 'rotary' if use_rope else 'none'
        use_softmax = False
        self.low_rank_kernel_x = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,  # add a diagonal bias
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)
        self.low_rank_kernel_y = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)
        self.to_out = nn.Sequential(
            MaskedInstanceNorm2d(),
            Rearrange('b c i l -> b i l c'),
            nn.Linear(2 * dim_head * heads, dim_out, bias=False),
            nn.GELU(),
            nn.Linear(dim_out, dim_out, bias=False))

    def forward(self, u_in, pos_lst, valid_mask):
        # x: b h w c
        if valid_mask.shape != u_in.shape[:-1]:
            raise ValueError("attention mask must have shape [B, H, W]")
        u = self.in_norm(u_in)
        v = self.to_v(u) * valid_mask.unsqueeze(-1)
        u = self.to_in(u)

        u_x = self.to_x[0](u, valid_mask)
        u_y = self.to_y[1](self.to_y[0](u), valid_mask.transpose(1, 2))
        pos_x, pos_y = pos_lst

        k_x = self.low_rank_kernel_x(u_x, pos_x=pos_x)
        k_y = self.low_rank_kernel_y(u_y, pos_x=pos_y)
        u_phi = rearrange(v, 'b i l (h c) -> b h i l c', h=self.heads)
        u_phi_x = torch.einsum('bhij,bhjlc->bhilc', k_x, u_phi)
        u_phi_y = torch.einsum('bhlm,bhimc->bhilc', k_y, u_phi)
        
        u_phi = torch.cat([u_phi_x, u_phi_y], dim=4)
        u_phi = rearrange(u_phi, 'b h i l c -> b (h c) i l', h=self.heads)
        
        u = self.to_out[1:](self.to_out[0](u_phi, valid_mask))
        
        return u
    
class FABlock2D_o(nn.Module):
    # contains factorization and attention on each axis
    def __init__(self,
                 dim,
                 dim_head,
                 latent_dim,
                 heads,
                 dim_out,
                 use_rope=True,
                 kernel_multiplier=3,
                 scaling_factor=1.0):
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.dim_head = dim_head
        self.in_norm = nn.LayerNorm(dim)
        self.to_v = nn.Linear(self.dim, heads * dim_head, bias=False)
        self.to_in = nn.Linear(self.dim, self.dim, bias=False)

        self.to_x = nn.Sequential(
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )
        self.to_y = nn.Sequential(
            Rearrange('b nx ny c -> b ny nx c'),
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )

        positional_encoding = 'rotary' if use_rope else 'none'
        use_softmax = False
        self.low_rank_kernel_x = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,  # add a diagonal bias
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)
        self.low_rank_kernel_y = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)
        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(dim_head * heads),
            Rearrange('b c i l -> b i l c'),
            nn.Linear(dim_head * heads, dim_out, bias=False),
            nn.GELU(),
            nn.Linear(dim_out, dim_out, bias=False))

    def forward(self, u, pos_lst):
        # x: b h w c
        u = self.in_norm(u)
        v = self.to_v(u)
        u = self.to_in(u)

        u_x = self.to_x(u)
        u_y = self.to_y(u)
        pos_x, pos_y = pos_lst

        k_x = self.low_rank_kernel_x(u_x, pos_x=pos_x)
        k_y = self.low_rank_kernel_y(u_y, pos_x=pos_y)
        u_phi = rearrange(v, 'b i l (h c) -> b h i l c', h=self.heads)
        u_phi = torch.einsum('bhij,bhjlc->bhilc', k_x, u_phi)
        u_phi = torch.einsum('bhlm,bhimc->bhilc', k_y, u_phi)
        u_phi = rearrange(u_phi, 'b h i l c -> b (h c) i l', h=self.heads)

        return self.to_out(u_phi)
