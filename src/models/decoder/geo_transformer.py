# Copyright (C) 2024-present Alibaba yuanjing aigclib Corporation. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

from .curope3d import cuRoPE3D
from .deformable_cross_attention import MSDCAWrapper


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    dropout: float = 0.1
    grid_size: int = 128

    max_batch_size: int = 32
    
    deform_att_cfg: dict = field(default_factory=dict)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

        self.rope3d = cuRoPE3D(freq=args.rope_theta)
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_p = args.dropout

    def forward(
        self,
        xyzs_idx: torch.Tensor,
        h: torch.Tensor,
    ):
        bsz, seqlen, _ = h.shape
        xq, xk, xv = self.wq(h), self.wk(h), self.wv(h)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq = self.rope3d(xq, xyzs_idx)
        xk = self.rope3d(xk, xyzs_idx)

        if xq.dtype == torch.float16 or xq.dtype == torch.bfloat16:
            qkv = torch.stack([xq, xk, xv], dim=2)
            output = flash_attn_qkvpacked_func(qkv=qkv, dropout_p=self.dropout_p if self.training else 0)
            output = output.view(bsz, seqlen, -1)
        
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = self.dropout(scores)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeoTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.self_attention = SelfAttention(args)
        self.cross_attention = MSDCAWrapper(
            dropout=args.dropout,
            **args.deform_att_cfg
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.self_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(
        self,
        xyzs_idx: torch.Tensor,
        latents: torch.Tensor,
        images: torch.Tensor,
        **kwargs,
    ):
        h = latents + self.dropout(self.self_attention(xyzs_idx, self.self_attention_norm(latents)))
        
        h = self.cross_attention(
            query=self.cross_attention_norm(h.float()),
            value=images.float(),
            **kwargs
        ).to(latents)
        
        out = h + self.dropout(self.feed_forward(self.ffn_norm(h)))

        return out


class GeoTransformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.grid_size = params.grid_size

        self.query_token = nn.Parameter(torch.randn(1, 1, params.dim - 3))

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(GeoTransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
    
    def project_points(self, xyzs, proj_matrix):
        B, S, _ = xyzs.shape
        V = proj_matrix.shape[1]
        
        # Project 3D points to image space
        pts3d = torch.cat([xyzs, torch.ones_like(xyzs[..., :1])], dim=-1)
        pts3d = pts3d.view(B, 1, S, 4, 1)
        proj_matrix = proj_matrix.view(B, V, 1, 3, 4)
        pts2d = torch.matmul(proj_matrix, pts3d).squeeze(-1)  # [B, V, S, 3]
        depth = pts2d[..., 2:]
        uvs = pts2d[..., :2] / depth  # [0, 1]
        uvs[..., 0] = 1. - uvs[..., 0]  # blender to opencv
        return uvs

    def forward(self, xyzs, img_feats, spatial_shapes, proj_matrix, xyz_scale=1.0, **kwargs):
        """Constructs 3D latents from input images and camera parameters.

        Args:
            xyzs: [B, S, 3], 3D coordinates of the input points
            img_feats: [B, V, L, C], input hierarchical image features.
                V: number of views
                L: h_0 * w_0 + h_1 * w_1 + ... + h_{num_levels-1} * w_{num_levels-1}
                C: feature dimension
                Refer to Deformable Cross-Attention for more details.
            spatial_shapes: [V, 2], spatial shapes of the hierarchical image features
            proj_matrix: [B, V, 3, 4], world-to-image projection matrices

        Returns:
            latents: [B, S, D], 3D latents of the input points
        """
        _bsz, seqlen, _ = xyzs.shape
        
        h = torch.cat([
            xyzs,
            self.query_token.repeat(_bsz, seqlen, 1).to(xyzs)
        ], dim=-1)
        
        xyzs_idx = (xyzs / xyz_scale + 1.0) * (self.grid_size - 1) / 2.0
        xyzs_idx = xyzs_idx.long()
        
        reference_points_cam = self.project_points(xyzs, proj_matrix)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))
        
        for layer in self.layers:
            h = layer(
                xyzs_idx=xyzs_idx,
                latents=h,
                images=img_feats,
                reference_points_cam=reference_points_cam,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )
        
        h = self.norm(h)
        
        return h
