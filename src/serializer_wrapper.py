# Copyright (C) 2024-present Yuanjing Shengsheng (Beijing) Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

import os
import math

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid, save_image
from torchvision.ops import sigmoid_focal_loss
import pytorch_lightning as pl
from pytorch_lightning.utilities.grads import grad_norm
from einops import rearrange, repeat
from torch_warmup_lr import WarmupLR

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_rays
from src.utils.loss_util import geo_scal_loss


class GeoSRL(pl.LightningModule):
    def __init__(
        self,
        serializer_config,
        input_size=448,
        warmup_steps=500,
    ):
        super(GeoSRL, self).__init__()

        self.input_size = input_size
        self.warmup_steps = warmup_steps
        # init modules
        self.serializer = instantiate_from_config(serializer_config).requires_grad_(True)
    
    def prepare_batch_data(self, batch):        
        images_orth = v2.functional.resize(
            batch['input_images'], self.input_size, interpolation=InterpolationMode.BILINEAR, antialias=True).clamp(0, 1)
        images_rand = v2.functional.resize(
            batch['target_images'], self.input_size, interpolation=InterpolationMode.BILINEAR, antialias=True).clamp(0, 1)

        # alphas_orth = v2.functional.resize(
        #     batch['input_alphas'], self.input_size, interpolation=0, antialias=True)
        # alphas_rand = v2.functional.resize(
        #     batch['target_alphas'], self.input_size, interpolation=0, antialias=True)
        
        # Note: Do not resize depth maps with bilinear interpolation,
        # otherwise the 3d points will contain lots of noise.
        depths_orth = batch['input_depths']
        depths_rand = batch['target_depths']

        c2ws_orth = batch['input_c2ws'].float()
        c2ws_rand = batch['target_c2ws'].float()
        Ks_orth = batch['input_Ks'].float()
        Ks_rand = batch['target_Ks'].float()

        # Sample input images
        input_dict = {}
        n_orth = torch.randint(1, 5, (1,))
        n_rand = torch.randint(0, 5, (1,))
        idx_orth = torch.randperm(4)[:n_orth].long()
        idx_rand = torch.randperm(4)[:n_rand].long()
        input_images = torch.cat([images_orth[:, idx_orth], images_rand[:, idx_rand]], dim=1).to(self.device)
        input_c2ws = torch.cat([c2ws_orth[:, idx_orth], c2ws_rand[:, idx_rand]], dim=1).to(self.device)
        input_Ks = torch.cat([Ks_orth[:, idx_orth], Ks_rand[:, idx_rand]], dim=1).to(self.device)
        input_dict = {
            'input_images': input_images,
            'input_c2ws': input_c2ws,
            'input_Ks': input_Ks,
        }
        
        if 'occupancy' in batch:
            gt_occupancy = batch['occupancy'].to(self.device)
        else:
            # Mix all depths to generate occupancy ground truth
            depths = torch.cat([depths_orth, depths_rand], dim=1).to(self.device)
            # alphas = torch.cat([alphas_orth, alphas_rand], dim=1).to(self.device)
            c2ws = torch.cat([c2ws_orth, c2ws_rand], dim=1).to(self.device)
            Ks = torch.cat([Ks_orth, Ks_rand], dim=1).to(self.device)
            gt_occupancy = self.get_occupancy(depths, c2ws, Ks)

        return input_dict, gt_occupancy
    
    def get_occupancy(self, depths, c2ws, Ks):
        B, V, _, h, w = depths.shape
        grid_size = self.serializer.grid_size
        Ks_ = Ks.clone()
        Ks_[:, :, 0] = Ks_[:, :, 0] * w
        Ks_[:, :, 1] = Ks_[:, :, 1] * h
        rays_o, rays_d = [], []
        for b in range(B):
            for v in range(V):
                rays_o_b, rays_d_b = get_rays(c2ws[b, v], h, w, Ks_[b, v, 0, 0])
                rays_o.append(rays_o_b)
                rays_d.append(rays_d_b)
        rays_o = torch.stack(rays_o)
        rays_d = torch.stack(rays_d)
        rays_o = rearrange(rays_o, '(b v) (h w) c -> b v c h w', b=B, v=V, h=h, w=w)
        rays_d = rearrange(rays_d, '(b v) (h w) c -> b v c h w', b=B, v=V, h=h, w=w)
        pts = rays_o + rays_d * depths
        pts = rearrange(pts, 'b v c h w -> b v h w c')
        masks = (depths > 0).squeeze(2)
        occs = pts.new_zeros((B, grid_size, grid_size, grid_size))
        for b in range(B):
            pts_masked_b = pts[b][masks[b]]
            pts_idx = torch.round((pts_masked_b + 0.5) * grid_size).long()
            pts_idx = torch.clamp(pts_idx, 0, grid_size - 1)
            occs[b, pts_idx[:, 0], pts_idx[:, 1], pts_idx[:, 2]] = 1
        return occs  # [B, X, Y, Z]
    
    def prepare_validation_batch_data(self, batch):
        lrm_generator_input = {}
        return lrm_generator_input
    
    def forward(self, input_dict):
        return self.serializer.pred_occupancy(**input_dict)

    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        scheduler.step()
        
        input_dict, gt_occupancy = self.prepare_batch_data(batch)
        out = self.forward(input_dict)
        loss, loss_dict = self.compute_loss(out, gt_occupancy)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def compute_loss(self, pred_occ, gt_occ):
        # loss_cls = sigmoid_focal_loss(pred_occ.flatten(), gt_occ.flatten(), alpha=0.95, gamma=2.0, reduction='mean')
        loss_cls = F.binary_cross_entropy_with_logits(pred_occ.flatten(), gt_occ.flatten())
        loss_geo = geo_scal_loss(pred_occ, gt_occ)
        tp = ((pred_occ > 0.5) & (gt_occ > 0.5)).sum()
        fp = ((pred_occ > 0.5) & (gt_occ < 0.5)).sum()
        fn = ((pred_occ < 0.5) & (gt_occ > 0.5)).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        iou = tp / (tp + fp + fn)
        loss = loss_cls + loss_geo
        # loss = loss_cls * 100
        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_cls': loss_cls})
        # loss_dict.update({f'{prefix}/loss_geo': loss_geo})
        loss_dict.update({f'{prefix}/loss': loss})
        loss_dict.update({f'{prefix}/iou': iou})
        loss_dict.update({f'{prefix}/precision': precision})
        loss_dict.update({f'{prefix}/recall': recall})
        loss_dict.update({f'{prefix}/tp': tp})
        loss_dict.update({f'{prefix}/fp': fp})
        loss_dict.update({f'{prefix}/fn': fn})
        return loss, loss_dict
    
    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self.serializer, norm_type=2)
    #     self.log_dict({'grad_norm/serializer': norms['grad_2.0_norm_total']})

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pass
    
    def on_validation_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        lr = self.learning_rate

        optimizer = torch.optim.AdamW(
            self.serializer.parameters(), lr=lr, betas=(0.90, 0.95), weight_decay=0.05)

        T_warmup, T_max, eta_min = self.warmup_steps, 50_000, 0.001
        lr_lambda = lambda step: \
            eta_min + (1 - math.cos(math.pi * step / T_warmup)) * (1 - eta_min) * 0.5 if step < T_warmup else \
            eta_min + (1 + math.cos(math.pi * (step - T_warmup) / (T_max - T_warmup))) * (1 - eta_min) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
