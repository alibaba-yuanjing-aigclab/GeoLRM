# Copyright (C) 2024-present Yuanjing Shengsheng (Beijing) Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

import os
import argparse
import numpy as np
import torch
import rembg
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.geolrm_wrapper import GeoLRM
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.infer_util import remove_background, resize_foreground, save_video


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
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
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

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
print('Loading custom white-background unet ...')
if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)

# load reconstruction model
print('Loading reconstruction model ...')
model = GeoLRM(**model_config['params'])
# new_srl_sd = torch.load('ckpts/srl-olgt-50k.ckpt', map_location='cpu')
# model.load_state_dict(new_srl_sd, strict=False)

model = model.to(device)
model = model.eval()

# make output directories
image_path = os.path.join(args.output_path, config_name, 'images')
# mesh_path = os.path.join(args.output_path, config_name, 'meshes')
gauss_path = os.path.join(args.output_path, config_name, 'gaussians')
video_path = os.path.join(args.output_path, config_name, 'videos')
os.makedirs(image_path, exist_ok=True)
# os.makedirs(mesh_path, exist_ok=True)
os.makedirs(gauss_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# process input files
if os.path.isdir(args.input_path):
    input_files = [
        os.path.join(args.input_path, file) 
        for file in os.listdir(args.input_path) 
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
    ]
else:
    input_files = [args.input_path]
print(f'Total number of input images: {len(input_files)}')


###############################################################################
# Stage 1: Multiview generation.
###############################################################################

rembg_session = None if args.no_rembg else rembg.new_session()

outputs = []
for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).split('.')[0]
    print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')

    # remove background optionally
    input_image = Image.open(image_file)
    if not args.no_rembg:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
    
    # sampling
    output_image = pipeline(
        input_image, 
        num_inference_steps=args.diffusion_steps, 
    ).images[0]

    output_image.save(os.path.join(image_path, f'{name}.png'))
    print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    outputs.append({'name': name, 'images': images})

# delete pipeline to save memory
del pipeline

###############################################################################
# Stage 2: Reconstruction.
###############################################################################

input_c2ws, input_Ks = get_zero123plus_input_cameras(batch_size=1, radius=1.5*args.scale, return_org=True)
input_c2ws, input_Ks = input_c2ws.to(device)[None], input_Ks.to(device)[None]

for idx, sample in tqdm(enumerate(outputs)):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 448, interpolation=3, antialias=True).clamp(0, 1)
    
    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_Ks = input_Ks[:, indices]
        input_c2ws = input_c2ws[:, indices]

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
