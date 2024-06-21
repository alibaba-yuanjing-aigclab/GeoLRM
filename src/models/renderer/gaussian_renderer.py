import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


class GaussianRenderer(nn.Module):
    def __init__(self,
                 transformer_dim=768,
                 gs_per_token=32,
                 use_sh=False,
                 offset_max=0.2,
                 scale_max=0.02):
        super(GaussianRenderer, self).__init__()
        # means3D, opacity, scale, rotation, sh(2-dog)/rgb
        if use_sh:
            gs_dim = 3 + 1 + 3 + 4 + 27
        else:
            gs_dim = 3 + 1 + 3 + 4 + 3
        self.use_sh = use_sh
        self.gs_dim = gs_dim
        out_dim = gs_dim * gs_per_token
        self.latent_decoder = nn.Sequential(
            nn.Linear(transformer_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.offset_max = offset_max
        self.scale_max = scale_max

    def get_gaussians(self, xyzs, latents):
        gaussian_features = self.latent_decoder(latents)
        
        gaussian_features = rearrange(gaussian_features, 'b n (k g) -> b n k g', g=self.gs_dim)
        mean3D_offsets = gaussian_features[..., :3].sigmoid() * self.offset_max
        gaussian_features[..., :3] = xyzs[:, :, None] + mean3D_offsets
        
        gaussian_features = rearrange(gaussian_features, 'b n k g -> b (n k) g', g=self.gs_dim)
        gaussian_features[:, :, 3:4] = gaussian_features[:, :, 3:4].sigmoid()  # opacity
        gaussian_features[:, :, 4:7] = gaussian_features[:, :, 4:7].sigmoid() * self.scale_max  # scale
        gaussian_features[:, :, 7:11] = F.normalize(gaussian_features[:, :, 7:11], dim=-1)  # rotation
        if self.use_sh:
            pass
        else:
            gaussian_features[:, :, 11:] = gaussian_features[:, :, 11:].clamp(-0.5, 0.5) + 0.5  # RGB
        return gaussian_features
        
    def forward(self, xyzs, latents, target_c2ws, target_Ks, bg_color, render_size, **kwargs):
        gaussian_features = self.get_gaussians(xyzs, latents)
        out = self.render(gaussian_features, target_c2ws, target_Ks, render_size=render_size, bg_color=bg_color)
        return out

    def cal_proj_matrix(self, tan_half_fovx, tan_half_fovy, znear, zfar):
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        proj_matrix[0, 0] = 1 / tan_half_fovx
        proj_matrix[1, 1] = 1 / tan_half_fovy
        proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
        proj_matrix[3, 2] = - (zfar * znear) / (zfar - znear)
        proj_matrix[2, 3] = 1
        return proj_matrix

    def render(self,
               gaussians,
               render_c2ws,
               render_Ks,
               render_size,
               bg_color=None,
               scale_modifier=1):
        # gaussians: [B, N, 14]
        # render_c2ws: [B, V, 4, 4]
        # render_Ks: [B, V, 3, 3]
        # bg_color: [3]
        # render_size

        device = gaussians.device
        B, V = render_c2ws.shape[:2]

        if bg_color is None:
            bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        else:
            bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)

        if isinstance(render_size, int):
            render_h, render_w = render_size, render_size
        else:
            render_h, render_w = render_size

        # loop of loop...
        images, alphas, depths = [], [], []
        for b in range(B):

            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            if self.use_sh:
                shs = gaussians[b, :, 11:].reshape(-1, 9, 3).contiguous().float()
                rgbs = None
            else:
                shs = None
                rgbs = gaussians[b, :, 11:].contiguous().float()

            for v in range(V):
                # render novel views
                K = render_Ks[b, v].float()
                # Here fx, fy are normalized by the image size
                fx, fy = K[0, 0], K[1, 1]
                tan_half_fovx = 1 / (2 * fx).float()
                tan_half_fovx = 1 / (2 * fy).float()
                
                transform = torch.diag(torch.tensor([1, -1, -1, 1], device=device).float())
                c2w = render_c2ws[b, v].float() @ transform  # to opencv camera space
                cam_view = torch.inverse(c2w).transpose(0, 1)
                cam_pos = - c2w[:3, 3]
                camera_dist = torch.norm(cam_pos)
                near = camera_dist - 0.867  # sqrt(3) * 0.5
                far = camera_dist + 0.867
                cam_view_proj = cam_view @ self.cal_proj_matrix(
                    tan_half_fovx, tan_half_fovx, near, far).to(device)

                raster_settings = GaussianRasterizationSettings(
                    image_height=render_h,
                    image_width=render_w,
                    tanfovx=tan_half_fovx,
                    tanfovy=tan_half_fovx,
                    bg=bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=cam_view,
                    projmatrix=cam_view_proj,
                    sh_degree=2 if self.use_sh else 0,
                    campos=cam_pos,
                    prefiltered=False,
                    debug=False,
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                # Rasterize visible Gaussians to image, obtain their radii (on screen).
                rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                    means3D=means3D,
                    means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                    shs=shs,
                    colors_precomp=rgbs,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )
                rendered_image = rendered_image.clamp(0, 1)

                images.append(rendered_image)
                alphas.append(rendered_alpha)
                depths.append(rendered_depth)

        images = torch.stack(images, dim=0).view(B, V, 3, render_h, render_w)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, render_h, render_w)
        depths = torch.stack(depths, dim=0).view(B, V, 1, render_h, render_w)
        
        return {
            "img": images, # [B, V, 3, H, W]
            "mask": alphas, # [B, V, 1, H, W]
            "depth": depths, # [B, V, H, W]
            "normal": None,
        }

    def save_ply(self, gaussians, path):
        # gaussians: [B, N, 14]
        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement

        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        return gaussians
