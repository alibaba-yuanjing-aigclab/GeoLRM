# Copyright (C) 2024-present Yuanjing Shengsheng (Beijing) Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

import os
import math
import json

import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid, save_image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import pytorch_lightning as pl
from pytorch_lightning.utilities.grads import grad_norm
from einops import rearrange, repeat

from src.utils.train_util import instantiate_from_config
from src.utils.loss_util import EdgeAwareLogL1, TVLoss


class GeoLRM(pl.LightningModule):
    def __init__(
        self,
        lrm_generator_config,
        serializer_config,
        input_size=256,
        render_size=512,
        init_ckpt=None,
        use_checkpoint=False,
        rand_views=True,
        warmup_steps=3000,
        max_steps=250_000,
        lambda_mse=1.0,
        lambda_lpips=2.0,
        lambda_mask=1.0,
        lambda_depth=0.2,
        lambda_depth_tv=0.05,
    ):
        super(GeoLRM, self).__init__()

        self.input_size = input_size
        self.render_size = render_size
        self.use_checkpoint = use_checkpoint
        self.rand_views = rand_views

        # init modules
        self.serializer = instantiate_from_config(serializer_config).requires_grad_(False)
        
        self.lrm_generator = instantiate_from_config(lrm_generator_config)

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.ssim = StructuralSimilarityIndexMeasure()
        self.depth_loss = EdgeAwareLogL1(implementation='scalar')
        self.tv_loss = TVLoss()
        
        self.lambda_mse = lambda_mse
        self.lambda_lpips = lambda_lpips
        self.lambda_mask = lambda_mask
        self.lambda_depth = lambda_depth
        self.lambda_depth_tv = lambda_depth_tv

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        if init_ckpt is not None:
            sd = torch.load(init_ckpt, map_location='cpu')
            self.load_state_dict(sd, strict=False)
            print(f'Loaded weights from {init_ckpt}')
        
        self.validation_step_outputs = []
        self.validation_metrics = []
    
    def on_fit_start(self):
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'images_val'), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, 'wandb'), exist_ok=True)
    
    def prepare_batch_data(self, batch):
        images_orth = v2.functional.resize(
            batch['input_images'], self.input_size, interpolation=InterpolationMode.BILINEAR, antialias=True).clamp(0, 1)
        images_rand = v2.functional.resize(
            batch['target_images'], self.input_size, interpolation=InterpolationMode.BILINEAR, antialias=True).clamp(0, 1)

        alphas_orth = v2.functional.resize(
            batch['input_alphas'], self.input_size, interpolation=InterpolationMode.NEAREST, antialias=True)
        alphas_rand = v2.functional.resize(
            batch['target_alphas'], self.input_size, interpolation=InterpolationMode.NEAREST, antialias=True)
        
        # Note: Do not resize depth maps with bilinear interpolation,
        # otherwise the 3d points will contain lots of noise.
        depths_orth = v2.functional.resize(
            batch['input_depths'], self.input_size, interpolation=InterpolationMode.NEAREST, antialias=True)
        depths_rand = v2.functional.resize(
            batch['target_depths'], self.input_size, interpolation=InterpolationMode.NEAREST, antialias=True)

        c2ws_orth = batch['input_c2ws'].float()
        c2ws_rand = batch['target_c2ws'].float()
        Ks_orth = batch['input_Ks'].float()
        Ks_rand = batch['target_Ks'].float()

        # Sample input images
        input_dict = {}
        n_orth = torch.randint(1, 5, (1,))
        n_rand = torch.randint(0, 4, (1,))
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
        
        # Mix all depths to generate occupancy ground truth
        images = torch.cat([images_orth, images_rand], dim=1).to(self.device)
        depths = torch.cat([depths_orth, depths_rand], dim=1).to(self.device)
        alphas = torch.cat([alphas_orth, alphas_rand], dim=1).to(self.device)
        images = v2.functional.resize(
            images, self.render_size, interpolation=InterpolationMode.BILINEAR, antialias=True).clamp(0, 1)
        depths = v2.functional.resize(
            depths, self.render_size, interpolation=InterpolationMode.NEAREST, antialias=True)
        alphas = v2.functional.resize(
            alphas, self.render_size, interpolation=InterpolationMode.NEAREST, antialias=True)
        c2ws = torch.cat([c2ws_orth, c2ws_rand], dim=1).to(self.device)
        Ks = torch.cat([Ks_orth, Ks_rand], dim=1).to(self.device)
        bg_color = torch.rand(3, device=self.device)
        bg = (1 - alphas).repeat(1, 1, 3, 1, 1) * bg_color[None, None, :, None, None].repeat(*images.shape[:2], 1, 1, 1)
        images = images * alphas + bg
        render_gt = {
            'target_images': images,
            'target_alphas': alphas,
            'target_depths': depths,
        }
        input_dict.update({
            'target_c2ws': c2ws,
            'target_Ks': Ks,
            'bg_color': bg_color
        })
        return input_dict, render_gt
    
    def prepare_validation_batch_data(self, batch):
        input_images = v2.functional.resize(
            batch['input_images'], self.input_size, interpolation=InterpolationMode.BILINEAR, antialias=True).clamp(0, 1)
        target_images = v2.functional.resize(
            batch['target_images'], self.input_size, interpolation=InterpolationMode.BILINEAR, antialias=True).clamp(0, 1)

        input_alphas = v2.functional.resize(
            batch['input_alphas'], self.input_size, interpolation=InterpolationMode.NEAREST, antialias=True)
        target_alphas = v2.functional.resize(
            batch['target_alphas'], self.input_size, interpolation=InterpolationMode.NEAREST, antialias=True)
        
        # Note: Do not resize depth maps with bilinear interpolation,
        # otherwise the 3d points will contain lots of noise.
        target_depths = v2.functional.resize(
            batch['target_depths'], self.input_size, interpolation=InterpolationMode.NEAREST, antialias=True)

        input_c2ws = batch['input_c2ws'].float()
        target_c2ws = batch['target_c2ws'].float()
        input_Ks = batch['input_Ks'].float()
        target_Ks = batch['target_Ks'].float()

        # Sample input images
        input_dict = {}
        input_images = input_images.to(self.device)
        input_c2ws = input_c2ws.to(self.device)
        input_Ks = input_Ks.to(self.device)
        input_dict = {
            'input_images': input_images,
            'input_c2ws': input_c2ws,
            'input_Ks': input_Ks,
        }
        
        # Mix all depths to generate occupancy ground truth
        target_images = target_images.to(self.device)
        target_depths = target_depths.to(self.device)
        target_alphas = target_alphas.to(self.device)
        target_images = v2.functional.resize(
            target_images, self.render_size, interpolation=InterpolationMode.BILINEAR, antialias=True).clamp(0, 1)
        target_depths = v2.functional.resize(
            target_depths, self.render_size, interpolation=InterpolationMode.NEAREST, antialias=True)
        target_alphas = v2.functional.resize(
            target_alphas, self.render_size, interpolation=InterpolationMode.NEAREST, antialias=True)
        target_c2ws = target_c2ws.to(self.device)
        target_Ks = target_Ks.to(self.device)
        bg_color = torch.ones(3, device=self.device)
        render_gt = {
            'target_images': target_images,
            'target_alphas': target_alphas,
            'target_depths': target_depths,
        }
        input_dict.update({
            'target_c2ws': target_c2ws,
            'target_Ks': target_Ks,
            'bg_color': bg_color
        })
        return input_dict, render_gt
    
    def forward(self, input_dict):
        xyzs, _ = self.serializer(**input_dict)
        xyzs = xyzs.clone().float()  # make differentiable
        
        if self.use_checkpoint:
            latents = torch.utils.checkpoint.checkpoint(
                self.lrm_generator.forward_latents,
                xyzs,
                **input_dict,
                xyz_scale=self.serializer.xyz_scale,
                use_reentrant=False
            )
        else:
            latents = self.lrm_generator.forward_latents(
                xyzs,
                **input_dict,
                xyz_scale=self.serializer.xyz_scale,
            )

        out = self.lrm_generator.renderer(
            xyzs,
            latents,
            **input_dict,
            render_size=self.render_size,
        )
        
        return out

    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        scheduler.step()
        
        input_dict, render_gt = self.prepare_batch_data(batch)

        render_out = self.forward(input_dict)

        loss, loss_dict = self.compute_loss(render_out, render_gt)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.global_step % 200 == 0 and self.global_rank == 0:
            target_images = rearrange(
                render_gt['target_images'], 'b n c h w -> b c h (n w)')
            render_images = rearrange(
                render_out['img'], 'b n c h w -> b c h (n w)')
            target_alphas = rearrange(
                repeat(render_gt['target_alphas'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            render_alphas = rearrange(
                repeat(render_out['mask'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            target_depths = rearrange(
                repeat(render_gt['target_depths'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            render_depths = rearrange(
                repeat(render_out['depth'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
            # target_normals = rearrange(
            #     render_gt['target_normals'], 'b n c h w -> b c h (n w)')
            # render_normals = rearrange(
            #     render_out['normal'], 'b n c h w -> b c h (n w)')
            MAX_DEPTH = torch.max(target_depths)
            target_depths = target_depths / MAX_DEPTH * target_alphas
            render_depths = render_depths / MAX_DEPTH

            grid = torch.cat([
                target_images, render_images, 
                target_alphas, render_alphas, 
                target_depths, render_depths, 
                # target_normals, render_normals,
            ], dim=-2)
            grid = make_grid(grid, nrow=target_images.shape[0], normalize=True, value_range=(0, 1))
            # self.logger.log_image('train/render', [grid], step=self.global_step)
            image_path = os.path.join(self.logdir, 'images', f'train_{self.global_step:07d}.png')
            save_image(grid, image_path)
            print(f"Saved image to {image_path}")

        return loss
    
    def compute_loss(self, render_out, render_gt):
        # NOTE: the rgb value range of OpenLRM is [0, 1]
        render_images = render_out['img']
        target_images = render_gt['target_images'].to(render_images)

        render_alphas = render_out['mask']
        target_alphas = render_gt['target_alphas']
        loss_mask = F.mse_loss(render_alphas, target_alphas)

        render_depths = render_out['depth']
        target_depths = render_gt['target_depths']
        mask = target_alphas > 0
        loss_depth = self.depth_loss(render_depths, target_depths, target_images, mask)

        render_images = rearrange(render_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        target_images = rearrange(target_images, 'b n ... -> (b n) ...') * 2.0 - 1.0
        loss_mse = F.mse_loss(render_images, target_images)
        loss_lpips = self.lpips(render_images, target_images)
        
        loss = loss_mse * self.lambda_mse \
            + loss_lpips * self.lambda_lpips \
            + loss_mask * self.lambda_mask \
            + loss_depth * self.lambda_depth
        
        prefix = 'train'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_mse': loss_mse})
        loss_dict.update({f'{prefix}/loss_lpips': loss_lpips})
        loss_dict.update({f'{prefix}/loss_mask': loss_mask})
        loss_dict.update({f'{prefix}/loss_depth': loss_depth})
        loss_dict.update({f'{prefix}/loss': loss})

        if self.lambda_depth_tv > 0:
            loss_depth_tv = self.tv_loss(render_depths) * self.lambda_depth_tv
            loss += loss_depth_tv
            loss_dict.update({f'{prefix}/loss_depth_tv': loss_depth_tv})

        return loss, loss_dict
    
    def compute_metrics(self, render_out, render_gt):
        render_images = rearrange(render_out['img'], 'b n ... -> (b n) ...')
        target_images = rearrange(render_gt['target_images'], 'b n ... -> (b n) ...').to(render_images)

        mse = F.mse_loss(render_images, target_images).mean()
        psnr = 10 * torch.log10(1.0 / mse)
        ssim = self.ssim(render_images, target_images)
        
        render_images = render_images * 2.0 - 1.0
        target_images = target_images * 2.0 - 1.0
        
        lpips = self.lpips(render_images, target_images)

        metrics = {
            'val/mse': mse,
            'val/pnsr': psnr,
            'val/ssim': ssim,
            'val/lpips': lpips,
        }
        return metrics

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.lrm_generator, norm_type=2)
        if 'grad_2.0_norm_total' in norms:
            self.log_dict({'grad_norm/lrm_generator': norms['grad_2.0_norm_total']})

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        input_dict, render_gt = self.prepare_validation_batch_data(batch)

        render_out = self.forward(input_dict)
        
        metrics = self.compute_metrics(render_out, render_gt)
        self.validation_metrics.append(metrics)
        
        render_images = render_out['img']
        render_images = rearrange(render_images, 'b n c h w -> b c h (n w)')
        gt_images = render_gt['target_images']
        gt_images = rearrange(gt_images, 'b n c h w -> b c h (n w)')
        log_images = torch.cat([render_images, gt_images], dim=-2)
        self.validation_step_outputs.append(log_images)
    
    def on_validation_epoch_end(self):
        images = torch.cat(self.validation_step_outputs, dim=-1)

        all_images = self.all_gather(images)
        all_images = rearrange(all_images, 'r b c h w -> (r b) c h w')

        if self.global_rank == 0:
            image_path = os.path.join(self.logdir, 'images_val', f'val_{self.global_step:07d}.png')

            grid = make_grid(all_images, nrow=1, normalize=True, value_range=(0, 1))
            save_image(grid, image_path)
            print(f"Saved image to {image_path}")

            metrics = {}
            for key in self.validation_metrics[0].keys():
                metrics[key] = torch.stack([m[key] for m in self.validation_metrics]).mean()
            self.log_dict(metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.validation_step_outputs.clear()
        self.validation_metrics.clear()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        input_dict, render_gt = self.prepare_validation_batch_data(batch)
        render_out = self.forward(input_dict)
        
        # Compute metrics
        metrics = self.compute_metrics(render_out, render_gt)
        self.validation_metrics.append(metrics)
        
        # Save images
        target_images = rearrange(
            render_gt['target_images'], 'b n c h w -> b c h (n w)')
        render_images = rearrange(
            render_out['img'], 'b n c h w -> b c h (n w)')
        target_alphas = rearrange(
            repeat(render_gt['target_alphas'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
        render_alphas = rearrange(
            repeat(render_out['mask'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
        target_depths = rearrange(
            repeat(render_gt['target_depths'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
        render_depths = rearrange(
            repeat(render_out['depth'], 'b n 1 h w -> b n 3 h w'), 'b n c h w -> b c h (n w)')
        # target_normals = rearrange(
        #     render_gt['target_normals'], 'b n c h w -> b c h (n w)')
        # render_normals = rearrange(
        #     render_out['normal'], 'b n c h w -> b c h (n w)')
        MAX_DEPTH = torch.max(target_depths)
        target_depths = target_depths / MAX_DEPTH * target_alphas
        render_depths = render_depths / MAX_DEPTH

        grid = torch.cat([
            target_images, render_images, 
            target_alphas, render_alphas, 
            target_depths, render_depths, 
            # target_normals, render_normals,
        ], dim=-2)
        grid = make_grid(grid, nrow=target_images.shape[0], normalize=True, value_range=(0, 1))
        # self.logger.log_image('train/render', [grid], step=self.global_step)
        image_path = os.path.join(self.logdir, 'images_test', f'{batch_idx:07d}.png')
        save_image(grid, image_path)
        print(f"Saved image to {image_path}")
    
    def on_test_start(self):
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, 'images_test'), exist_ok=True)
    
    def on_test_epoch_end(self):
        metrics = {}
        for key in self.validation_metrics[0].keys():
            metrics[key] = torch.stack([m[key] for m in self.validation_metrics]).cpu().numpy().tolist()
        metric_path = os.path.join(self.logdir, f'metrics.json')
        with open(metric_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to {metric_path}")
        
        for key in metrics.keys():
            metrics[key] = torch.stack([m[key] for m in self.validation_metrics]).mean()
        print(metrics)
        
        self.validation_metrics.clear()
    
    def configure_optimizers(self):
        lr = self.learning_rate

        optimizer = torch.optim.AdamW(
            self.lrm_generator.parameters(), lr=lr, betas=(0.90, 0.95), weight_decay=0.05)

        T_warmup, T_max, eta_min = self.warmup_steps, self.max_steps, 0.001
        lr_lambda = lambda step: \
            eta_min + (1 - math.cos(math.pi * step / T_warmup)) * (1 - eta_min) * 0.5 if step < T_warmup else \
            eta_min + (1 + math.cos(math.pi * (step - T_warmup) / (T_max - T_warmup))) * (1 - eta_min) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
