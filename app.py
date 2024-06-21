# Copyright (C) 2024-present Yuanjing Shengsheng (Beijing) Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

import os
import math
import tempfile

import cv2
import numpy as np
import torch
import imageio
from rembg import remove
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from typing import List, Optional
from torchvision.transforms import ToTensor
from einops import rearrange, repeat

from src.geolrm_wrapper import GeoLRM
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_sv3d_input_cameras,
    get_circular_camera_poses,
)
from src.utils.infer_util import save_video
from sgm.util import instantiate_from_config


def get_render_cameras(batch_size=1, M=120, radius=1.5, elevation=20.0):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    Ks = FOV_to_intrinsics(39.6).unsqueeze(0).repeat(M, 1, 1).float()
    c2ws = c2ws[None].repeat(batch_size, 1, 1, 1)
    Ks = Ks[None].repeat(batch_size, 1, 1, 1)
    return c2ws, Ks


config_path = 'configs/geolrm.yaml'
config = OmegaConf.load('configs/geolrm.yaml')
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

device = torch.device('cuda')

# load reconstruction model
print('Loading reconstruction model ...')
model = GeoLRM(**model_config['params'])

model = model.to(device)
model = model.eval()

# make output directories
output_path = 'tmp_gradio'
image_path = os.path.join(output_path, config_name, 'images')
# mesh_path = os.path.join(output_path, config_name, 'meshes')
gauss_path = os.path.join(output_path, config_name, 'gaussians')
video_path = os.path.join(output_path, config_name, 'videos')
sv3d_path = os.path.join(output_path, config_name, 'sv3d')
os.makedirs(image_path, exist_ok=True)
# os.makedirs(mesh_path, exist_ok=True)
os.makedirs(gauss_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)
os.makedirs(sv3d_path, exist_ok=True)

# SV3D model
sv3d_config = OmegaConf.load('configs/sv3d_p.yaml')
sv3d_config.model.params.conditioner_config.params.emb_models[0]\
    .params.open_clip_embedding_config.params.init_device = 'cuda'
sv3d_config.model.params.sampler_config.params.verbose = False
sv3d_config.model.params.sampler_config.params.num_steps = 50
sv3d_config.model.params.sampler_config.params.guider_config.params.num_frames = 21
sv3d_model = instantiate_from_config(sv3d_config.model).to(device).eval()


# Preprocess
def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(image, do_remove_background):
    if do_remove_background:
        image.thumbnail([768, 768], Image.Resampling.LANCZOS)
        image = remove(image.convert("RGBA"), alpha_matting=True)

    # resize object in frame
    image_arr = np.array(image)
    in_w, in_h = image_arr.shape[:2]
    ret, mask = cv2.threshold(
        np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
    )
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = in_w
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center - h // 2 : center - h // 2 + h,
        center - w // 2 : center - w // 2 + w,
    ] = image_arr[y : y + h, x : x + w]
    # resize frame to 576x576
    rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
    # white bg
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    return input_image


def sample_videos(
    processed_image, sample_seed,
    num_frames: Optional[int] = 21,
    version: str = "sv3d_p",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 1e-5,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = sv3d_path,
    elevations_deg: Optional[float | List[float]] = 10.0,  # For SV3D
    azimuths_deg: Optional[List[float]] = None,  # For SV3D
):
    seed_everything(sample_seed)
    
    if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
        elevations_deg = [elevations_deg] * num_frames
    assert (
        len(elevations_deg) == num_frames
    ), f"Please provide 1 value, or a list of {num_frames} values for elevations_deg! Given {len(elevations_deg)}"
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    if azimuths_deg is None:
        azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    assert (
        len(azimuths_deg) == num_frames
    ), f"Please provide a list of {num_frames} values for azimuths_deg! Given {len(azimuths_deg)}"
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()
    
    model = sv3d_model
    
    image = ToTensor()(processed_image)
    image = image * 2.0 - 1.0

    image = image.unsqueeze(0).to(device)
    H, W = image.shape[2:]
    print(f"Image shape: {image.shape}")
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)
    if (H, W) != (576, 1024) and "sv3d" not in version:
        print(
            "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
        )
    if (H, W) != (576, 576) and "sv3d" in version:
        print(
            "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as model was only trained on 576x576."
        )
    if motion_bucket_id > 255:
        print(
            "WARNING: High motion bucket! This may lead to suboptimal performance."
        )

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")

    value_dict = {}
    value_dict["cond_frames_without_noise"] = image
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
    if "sv3d_p" in version:
        value_dict["polars_rad"] = polars_rad
        value_dict["azimuths_rad"] = azimuths_rad

    input_img_name = tempfile.mktemp()
    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            randn = torch.randn(shape, device=device)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model, input, sigma, c, **additional_model_inputs
                )

            samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            if "sv3d" in version:
                samples_x[-1:] = value_dict["cond_frames_without_noise"]
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            os.makedirs(output_folder, exist_ok=True)
            
            save_path = os.path.join(output_folder, f"{input_img_name}")
            
            imageio.imwrite(
                save_path + ".png", processed_image
            )
            
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            vid_path = save_path + ".mp4"
            imageio.mimwrite(vid_path, vid)
            
    return vid_path


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def video_to_tensor(video_path):
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    video_np = np.array(frames)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
    return video_tensor


def make3d(sv3d_video, num_gaussians):
    model.serializer.max_num_points = num_gaussians
    name = os.path.basename(sv3d_video).split('.')[0]
    
    images = video_to_tensor(sv3d_video).unsqueeze(0).to(device) / 255.0
    images = v2.functional.resize(images, 560, interpolation=3, antialias=True).clamp(0, 1)
    
    input_c2ws, input_Ks = get_sv3d_input_cameras(batch_size=1, radius=1.5, return_org=True)
    input_c2ws, input_Ks = input_c2ws.to(device)[None], input_Ks.to(device)[None]
    
    step = 1
    indices = torch.arange(0, 21, step).long().to(device)
    images = images[:, indices].contiguous().clone()
    input_Ks = input_Ks[:, indices].contiguous().clone()
    input_c2ws = input_c2ws[:, indices].contiguous().clone()
    print(f"Images shape: {images.shape}")
    with torch.no_grad():
        # get latents
        xyzs, _ = model.serializer(images, input_c2ws, input_Ks)
        latents = model.lrm_generator.forward_latents(xyzs, images, input_Ks, input_c2ws)

        # get gaussians
        gaussians = model.lrm_generator.renderer.get_gaussians(xyzs, latents)
        ply_path = os.path.join(gauss_path, f'{name}.ply')
        model.lrm_generator.renderer.save_ply(gaussians, ply_path)
        
        # get video
        video_path_idx = os.path.join(video_path, f'{name}.mp4')
        render_size = infer_config.render_resolution
        render_c2ws, render_Ks = get_render_cameras(
            batch_size=1, 
            M=120,
            radius=1.5, 
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
    
    return video_path_idx, ply_path


import gradio as gr

_HEADER_ = '''
<h2><b>Official 🤗 Gradio Demo</b></h2><h2><a href='https://github.com/alibaba-yuanjing-aigclab/GeoLRM' target='_blank'><b>GeoLRM: Geometry-Aware Large Reconstruction Model for High-Quality 3D Gaussian Generation</b></a></h2>

<h3>
Tips for better results:

- Use high-resolution images for better results.
- Orthographic front-facing images lead to good reconstructions.
- Avoid white objects and overexposed images.
</h3>
'''

_LINKS_ = '''
<h3>Code is available at <a href='https://github.com/alibaba-yuanjing-aigclab/GeoLRM' target='_blank'>GitHub</a></h3>
<h3>Paper is available at <a href='https://arxiv.org/abs/TODO' target='_blank'>arXiv</a></h3>
'''
# TODO
_CITE_ = r"""
```bibtex
```
"""

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    width=256,
                    height=256,
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(
                    label="Processed Image", 
                    image_mode="RGB", 
                    width=256,
                    height=256,
                    type="pil", 
                    interactive=False
                )
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=False
                    )
                    sample_seed = gr.Number(value=42, label="Seed Value", precision=0)
                    num_gaussians = gr.Slider(
                        label="Number of Gaussians",
                        minimum=4096,
                        maximum=16384,
                        value=8192,
                        step=4096,
                    )

            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")

            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[
                        os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
                    ],
                    inputs=[input_image],
                    label="Examples",
                    examples_per_page=20
                )

        with gr.Column():

            with gr.Row():

                with gr.Column():
                    with gr.Tab(label="GeoLRM result"):
                        output_video = gr.Video(
                            label="Rendered 3D model", format="mp4",
                            width=379,
                            autoplay=True,
                            interactive=False
                        )
                    with gr.Tab(label="SV3D result"):
                        sv3d_video = gr.Video(
                            label="SV3D video", format="mp4",
                            width=379,
                            autoplay=True,
                            interactive=False
                        )
                
                with gr.Column():
                    gaussians = gr.File(
                        label="3D Gaussians (PLY Format)",
                        type="file",
                        width=379,
                        download=True,
                        elem_id="gaussians",
                        interactive=False
                    )
            
            with gr.Row():
                gr.Markdown('''Try a different <b>seed value</b> if the result is unsatisfying (Default: 42).''')

    gr.Markdown(_LINKS_)
    gr.Markdown(_CITE_)

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[processed_image],
    ).success(
        fn=sample_videos,
        inputs=[processed_image, sample_seed],
        outputs=[sv3d_video],
    ).success(
        fn=make3d,
        inputs=[sv3d_video, num_gaussians],
        outputs=[output_video, gaussians]
    )

demo.queue(max_size=10)
demo.launch(server_name="0.0.0.0", server_port=42339)
