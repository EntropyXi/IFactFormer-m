import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from libs.factorization_module import FABlock2D_m, FABlock2D_o

from libs.positional_encoding_module import GaussianFourierFeatureTransform
    

class FactorizedTransformer(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 dim_out,
                 depth,
                 n_layer,
                 model,
                 **kwargs
             ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            layer = nn.ModuleList([])
            layer.append(nn.Sequential(
                GaussianFourierFeatureTransform(3, dim // 2, 8),
                nn.Linear(dim, dim)
            ))
            if model=="IFactFormer_o":
                layer.append(FABlock2D_o(dim, dim_head, dim, heads, dim_out, use_rope=True, **kwargs))
            elif model=="IFactFormer_m":
                layer.append(FABlock2D_m(dim, dim_head, dim, heads, dim_out, use_rope=True, **kwargs))
            self.layers.append(layer)
            
        self.n_layer = n_layer
        self.mask_ocean = model == "IFactFormer_m"

    def forward(self, u, positions, valid_mask=None):
        b, nx, ny, c = u.shape  # just want to make sure its shape
        absolute = positions["absolute"]
        if absolute.shape != (b, nx, ny, 3):
            raise ValueError("absolute positions must have shape [B, H, W, 3]")
        pos = rearrange(absolute, 'b nx ny c -> b (nx ny) c')
        pos_lst = (positions["axis_lat"], positions["axis_lon"])
        if self.mask_ocean:
            if valid_mask is None or valid_mask.shape != (b, nx, ny):
                raise ValueError("IFactFormer_m requires an ocean mask [B, H, W]")
            spatial_mask = valid_mask.unsqueeze(-1)
        
        for l, (pos_enc, attn_layer) in enumerate(self.layers):
            u = u + rearrange(pos_enc(pos), 'b (nx ny) c -> b nx ny c', nx=nx, ny=ny)
            if self.mask_ocean:
                u = u * spatial_mask
            for i in range(self.n_layer):
                if self.mask_ocean:
                    u = (u + attn_layer(u, pos_lst, valid_mask) / self.n_layer) * spatial_mask
                else:
                    u = u + attn_layer(u, pos_lst) / self.n_layer
        return u
        
        
        
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.to_in = nn.Sequential(
            nn.Conv2d(config.in_dim, config.dim // 2, kernel_size=1, stride=1, padding=0, groups=config.in_dim),
            nn.GELU(),
            nn.Conv2d(config.dim // 2, config.dim, kernel_size=(config.in_time_window, 1), stride=1, padding=0, bias=False),
        )

        self.encoder = FactorizedTransformer(config.dim, config.dim_head, config.heads, config.dim, config.depth, config.n_layer, config.model)
        
        
        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny c -> b c (nx ny)'),
            nn.Conv1d(config.dim, config.dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv1d(config.dim // 2, config.out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )       
        
    def forward(self,
                u,
                positions,
                ):
        b, t, nx, ny, c = u.shape
        if self.encoder.mask_ocean:
            if c < 3:
                raise ValueError("IFactFormer_m requires U, V, and ocean-mask input channels")
            valid_mask = (u[..., 2] > 0.5).all(dim=1)
        else:
            valid_mask = None
        
        u = rearrange(u, 'b t nx ny c -> b c t (nx ny)')
        u = self.to_in(u)
        u = rearrange(u, 'b c 1 (nx ny) -> b nx ny c', nx=nx, ny=ny)
        if valid_mask is not None:
            u = u * valid_mask.unsqueeze(-1)
        
        u = self.encoder(u, positions, valid_mask)
        
        u = self.simple_to_out(u)
        u = rearrange(u, 'b c (nx ny) -> b nx ny c', nx=nx, ny=ny)
        
        return u
