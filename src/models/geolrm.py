# Copyright (C) 2024-present Yuanjing Shengsheng (Beijing) Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

import torch
import torch.nn as nn

from .encoder.dino2_wrapper import DinoWrapper
from .decoder.geo_transformer import ModelArgs, GeoTransformer
from .renderer.gaussian_renderer import GaussianRenderer


class GeoLRM(nn.Module):
    """
    Full model of the Geometry-aware Large Reconstruction Model.
    """
    def __init__(
        self,
        use_bf16=True,
        encoder_freeze: bool = False, 
        encoder_model_name: str = 'facebook/dinov2-base',
        encoder_feat_dim: int = 768,
        transformer_dim: int = 768,
        transformer_layers: int = 12,
        transformer_heads: int = 16,
        grid_size: int = 128,
        num_deform_points: int = 8,
        gs_per_token: int = 32,
        use_sh: bool = True,
        offset_max = 0.05,
        scale_max = 0.02,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_bf16 = use_bf16

        # modules
        self.encoder = DinoWrapper(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            drop_cls_token=True,
            out_dim=encoder_feat_dim
        )

        self.transformer = GeoTransformer(ModelArgs(
            dim=transformer_dim,
            n_layers=transformer_layers,
            n_heads=transformer_heads,
            n_kv_heads=transformer_heads,
            grid_size=grid_size,
            dropout=dropout,
            deform_att_cfg = dict(
                query_dim = transformer_dim,
                deformable_attention = dict(
                    type = 'MSDeformableAttention3D',
                    query_dim = transformer_dim,
                    value_dim = encoder_feat_dim,
                    num_heads = transformer_heads,
                    num_levels = 2,
                    num_points = num_deform_points,
                ))
        ))
        
        self.renderer = GaussianRenderer(
            transformer_dim=transformer_dim,
            gs_per_token=gs_per_token,
            use_sh=use_sh,
            offset_max=offset_max,
            scale_max=scale_max,
        )

    def forward_latents(self, xyzs, input_images, input_Ks, input_c2ws, **kwargs):
        B, V, C, H, W = input_images.shape

        with torch.cuda.amp.autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            # encode images
            image_feats, spatial_shapes = self.encoder(input_images, input_Ks, input_c2ws)
            
            input_w2cs = torch.inverse(input_c2ws)
            proj_matrix = input_Ks @ input_w2cs[:, :, :3, :4]
            
            latents = self.transformer(
                xyzs,
                image_feats,
                spatial_shapes,
                proj_matrix,
                **kwargs
            )

        return latents
    
    def forward(self, xyzs, images, input_Ks, input_c2ws, render_c2ws, render_Ks, bg_color, render_size: int):
        latents = self.forward_latents(xyzs, images, input_Ks, input_c2ws)
        out = self.renderer(xyzs, latents, render_c2ws, render_Ks, bg_color, render_size)
        return {
            'latents': latents,
            **out
        }

    def extract_mesh(
        self, 
        planes: torch.Tensor, 
        use_texture_map: bool = False,
        texture_resolution: int = 1024,
        **kwargs,
    ):
        pass
