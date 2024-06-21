# Copyright (C) 2024-present Yuanjing Shengsheng (Beijing) Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

import os
import argparse

import torch
from tqdm import tqdm
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm

from src.geolrm_wrapper import GeoLRM
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_sv3d_input_cameras,
    get_circular_camera_poses,
)
from src.utils.infer_util import save_video
from sv3d_video_sample import sample as sample_videos

def get_render_cameras(batch_size=1, M=120, radius=1.5, elevation=20.0):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    Ks = FOV_to_intrinsics(39.6).unsqueeze(0).repeat(M, 1, 1).float()
    c2ws = c2ws[None].repeat(batch_size, 1, 1, 1)
    Ks = Ks[None].repeat(batch_size, 1, 1, 1)
    return c2ws, Ks


###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=1.5, help='Render distance.')
parser.add_argument('--view', type=int, default=21, help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
# parser.add_argument('--export_mesh', action='store_true', help='Export a mesh.')
args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = False

device = torch.device('cuda')

# load reconstruction model
print('Loading reconstruction model ...')
model = GeoLRM(**model_config['params'])

model = model.to(device)
model = model.eval()

# make output directories
image_path = os.path.join(args.output_path, config_name, 'images')
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
gauss_path = os.path.join(args.output_path, config_name, 'gaussians')
video_path = os.path.join(args.output_path, config_name, 'videos')
sv3d_path = os.path.join(args.output_path, config_name, 'sv3d')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(gauss_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

###############################################################################
# Stage 1: Multiview generation.
###############################################################################

outputs = sample_videos(
    input_path=args.input_path,
    version="sv3d_p",
    elevations_deg=10.0,
    output_folder=sv3d_path,
)

###############################################################################
# Stage 2: Reconstruction.
###############################################################################

for idx, sample in tqdm(enumerate(outputs)):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 560, interpolation=3, antialias=True).clamp(0, 1)
    
    input_c2ws, input_Ks = get_sv3d_input_cameras(batch_size=1, radius=1.5*args.scale, return_org=True)
    input_c2ws, input_Ks = input_c2ws.to(device)[None], input_Ks.to(device)[None]
    
    step = 21 // args.view
    indices = torch.arange(0, 21, step).long().to(device)
    images = images[:, indices].contiguous().clone()
    input_Ks = input_Ks[:, indices].contiguous().clone()
    input_c2ws = input_c2ws[:, indices].contiguous().clone()
    
    with torch.no_grad():
        # get latents
        xyzs, _ = model.serializer(images, input_c2ws, input_Ks)
        latents = model.lrm_generator.forward_latents(xyzs, images, input_Ks, input_c2ws)

        # get gaussians
        gaussians = model.lrm_generator.renderer.get_gaussians(xyzs, latents)
        model.lrm_generator.renderer.save_ply(gaussians, os.path.join(gauss_path, f'{name}.ply'))
        
        # get video
        video_path_idx = os.path.join(video_path, f'{name}.mp4')
        render_size = infer_config.render_resolution
        render_c2ws, render_Ks = get_render_cameras(
            batch_size=1, 
            M=120,
            radius=args.distance, 
            elevation=20.0
        )
        render_c2ws, render_Ks = render_c2ws.to(device), render_Ks.to(device)
        
        # if args.export_mesh:
        #     mesh_path_idx = os.path.join(mesh_path, f'{name}.ply')
        #     model.lrm_generator.renderer.extract_mesh(
        #         gaussians[0],
        #         mesh_path_idx
        #     )
        out = model.lrm_generator.renderer.render(
            gaussians, 
            render_c2ws,
            render_Ks,
            render_size=render_size
        )
        frames = out["img"][0]

        save_video(
            frames,
            video_path_idx,
            fps=30,
        )
        print(f"Video saved to {video_path_idx}")
