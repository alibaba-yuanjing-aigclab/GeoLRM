# Copyright (C) 2024-present Yuanjing Shengsheng (Beijing) Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

from typing import Optional

import torch
from torch import nn
from einops import rearrange, repeat

from src.models.encoder.dino2_wrapper import DinoWrapper
from src.models.decoder.geo_transformer import ModelArgs, GeoTransformer


class TransformerSerializer(nn.Module):
    def __init__(
        self,
        use_bf16: bool = True,
        encoder_freeze: bool = False,
        encoder_model_name: str = 'facebook/dinov2-base',
        encoder_feat_dim: int = 384,
        transformer_dim: int = 768,
        transformer_layers: int = 6,
        transformer_heads: int = 16,
        grid_size: int = 256,
        token_res: int = 16,
        num_deform_points: int = 8,
        scale: float = 0.5,
        max_seq_len: int = 4096,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.encoder = DinoWrapper(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            out_dim=encoder_feat_dim
        )
        
        self.transformer = GeoTransformer(ModelArgs(
            dim=transformer_dim,
            n_layers=transformer_layers,
            n_heads=transformer_heads,
            n_kv_heads=transformer_heads,
            grid_size=grid_size,
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

        self.use_bf16 = use_bf16
        self.grid_size = grid_size
        self.token_res = token_res
        self.max_num_points = max_seq_len
        
        self.upsample = nn.Linear(transformer_dim, token_res ** 3)
        
        # Create a grid of 3D points
        self.xyz_scale = scale
        xyzs = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_size // token_res),
            torch.linspace(-1, 1, grid_size // token_res),
            torch.linspace(-1, 1, grid_size // token_res),
            indexing='ij',
        ), dim=-1) * scale
        self.xyzs = xyzs.view(-1, 3)
        
        xyzs_full = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, grid_size),
            torch.linspace(-1, 1, grid_size),
            torch.linspace(-1, 1, grid_size),
            indexing='ij',
        ), dim=-1) * scale
        self.xyzs_full = xyzs_full
        
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            sd = {k.replace('serializer.', ''): v for k, v in ckpt.items()}
            self.load_state_dict(sd, strict=True)
        
    def pred_occupancy(self, input_images, input_c2ws, input_Ks, **kwargs):
        # images: [B, V, C_img, H_img, W_img]
        B = input_images.shape[0]

        # encode images
        image_feats, spatial_shapes = self.encoder(input_images, input_Ks, input_c2ws)
        
        input_w2cs = torch.inverse(input_c2ws)
        proj_matrix = input_Ks @ input_w2cs[:, :, :3, :4]
        
        occs = self.transformer(
            self.xyzs[None].repeat(B, 1, 1).to(image_feats),
            image_feats,
            spatial_shapes,
            proj_matrix,
            xyz_scale=self.xyz_scale
        )
        
        f_res = self.grid_size // self.token_res
        t_res = self.token_res
        
        occs = self.upsample(occs)
        occs = occs.view(B, f_res, f_res, f_res, t_res, t_res, t_res)
        occs = occs.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        occs = occs.view(B, self.grid_size, self.grid_size, self.grid_size)
        return occs
    
    @torch.inference_mode()
    def forward(self, input_images, input_c2ws, input_Ks, **kwargs):
        B = input_images.shape[0]
        
        with torch.cuda.amp.autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            occs = self.pred_occupancy(input_images, input_c2ws, input_Ks)  # [B, X, Y, Z]
        
        xyzs_full = self.xyzs_full.to(occs)[None].repeat(B, 1, 1, 1, 1)
        
        # Serialize points
        max_num_points = self.max_num_points
        pts_list, mask_list = [], []
        for i in range(B):
            pts = xyzs_full[i][occs[i] > 0]
            if pts.shape[0] >= max_num_points:
                pts = pts[torch.randperm(pts.shape[0])[:max_num_points]]
                mask = torch.ones(max_num_points).to(pts.device, dtype=torch.bool)
            elif pts.shape[0] < max_num_points:
                mask = torch.cat([
                    torch.ones(pts.shape[0]),
                    torch.zeros(max_num_points - pts.shape[0])
                ], dim=0).to(device=pts.device, dtype=torch.bool)
                pts = torch.cat([
                    pts,
                    torch.zeros((max_num_points - pts.shape[0], 3)).to(pts)
                ], dim=0)
            pts_list.append(pts)
            mask_list.append(mask)
            
        pts = torch.stack(pts_list, dim=0)
        masks = torch.stack(mask_list, dim=0)
        
        return pts, masks
