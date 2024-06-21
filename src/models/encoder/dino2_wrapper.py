# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from transformers import Dinov2Backbone
from torchvision import transforms
from einops import rearrange

from src.utils.camera_util import get_rays


def get_ray_embedding(input_Ks, input_c2ws, h, w):
    """
    Get rays_o and rays_d from input Ks and c2ws.
    
    input_Ks: (B, 3, 3)
    input_c2ws: (B, 4, 4)
    """
    # The original Ks are scaled by the image size.
    input_Ks[:, 0] = input_Ks[:, 0] * w
    input_Ks[:, 1] = input_Ks[:, 1] * h
    B = input_Ks.shape[0]
    rays_o, rays_d = [], []
    for b in range(B):
        rays_o_b, rays_d_b = get_rays(input_c2ws[b], h, w, input_Ks[b, 0, 0])
        rays_o.append(rays_o_b)
        rays_d.append(rays_d_b)
    rays_o = torch.stack(rays_o)
    rays_d = torch.stack(rays_d)
    ray_embedding = torch.cat([
        rays_d, torch.cross(rays_o, rays_d, dim=-1)], dim=-1)
    return rearrange(ray_embedding, 'b (h w) c -> b c h w', h=h, w=w)


class DinoWrapper(nn.Module):
    """
    Dino v2 wrapper using huggingface transformer implementation.
    """
    def __init__(self,
                 model_name: str = 'facebook/dinov2-base',
                 freeze: bool = True,
                 drop_cls_token: bool = True,
                 out_dim=768):
        super().__init__()
        
        self.patch_size = 14
        
        self.model = Dinov2Backbone.from_pretrained(model_name)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dino_dim = self.model.encoder.config.hidden_size
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(dino_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        self.low_level_encoder = nn.Sequential(
            nn.Conv2d(9, out_dim, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
        )
        if freeze:
            self._freeze()
        self.drop_cls_token = drop_cls_token

    def forward(self, image, input_Ks, input_c2ws):
        # image: [B, N, C, H, W]
        # RGB image with [0,1] scale and properly sized
        if image.ndim == 5:
            _, N, _, H, W = image.shape
            mv = True
            image = image.flatten(0, 1)
            input_Ks_ = input_Ks.flatten(0, 1).clone()
            input_c2ws_ = input_c2ws.flatten(0, 1).clone()
        else:
            raise NotImplementedError
        
        inputs = self.normalize(image)
        outputs = self.model(inputs)
        feature_maps = outputs.feature_maps[-1]
        high_level_feature = self.deconv(feature_maps)
        
        ray_embedding = get_ray_embedding(input_Ks_, input_c2ws_, H, W)
        
        low_level_feature = self.low_level_encoder(torch.cat([inputs, ray_embedding], dim=1))
        
        if mv:
            b, c, hh, wh = high_level_feature.shape
            _, _, hl, wl = low_level_feature.shape
            hierarchical_img_feature = torch.cat([
                low_level_feature.reshape(b, c, -1),
                high_level_feature.reshape(b, c, -1),
            ], dim=-1)
            hierarchical_img_feature = rearrange(hierarchical_img_feature, '(b n) c l -> b n l c', n=N)
            spatial_shapes = torch.tensor([
                [hl, wl],
                [hh, wh],
            ], dtype=torch.long).to(image.device)
        
        return hierarchical_img_feature, spatial_shapes

    def _freeze(self):
        print(f"======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
