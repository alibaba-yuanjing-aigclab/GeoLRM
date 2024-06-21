# Copyright (C) 2024-present Yuanjing Shengsheng (Beijing) Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

import os
import json
import argparse

import cv2
import torch
import numpy as np
import open3d as o3d
from einops import rearrange
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.utils.camera_util import get_rays
from src.utils.camera_util import FOV_to_intrinsics

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


json_path = 'data/objaverse/gobjaverse_280k.json'
with open(json_path, 'r') as reader:
    objaverse_key = json.load(reader)
objaverse_key = sorted(objaverse_key)
data_root = 'data/objaverse/gobjaverse_280k'
occ_root = 'data/objaverse/gobjaverse_280k_occ'
GRID_SIZE = 128


def read_camera_info_single(json_file):
    with open(json_file, 'r', encoding='utf8') as reader:
        json_content = json.load(reader)

    c2w = np.eye(4)
    c2w[:3, 0] = np.array(json_content['x'])
    c2w[:3, 1] = np.array(json_content['y'])
    c2w[:3, 2] = np.array(json_content['z'])
    c2w[:3, 3] = np.array(json_content['origin'])
    # print(camera_matrix)
    
    fov_x = json_content['x_fov']
    fov_y = json_content['y_fov']

    return c2w, fov_x, fov_y


def get_occupancy(depths, c2ws, Ks):
    B, V, _, h, w = depths.shape
    grid_size = GRID_SIZE
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
    # occs = pts.new_zeros((B, grid_size, grid_size, grid_size))
    pts_list = []
    for b in range(B):
        pts_masked_b = pts[b][masks[b]]
        pts_idx = torch.round((pts_masked_b + 0.5) * grid_size)
        pts_idx = torch.unique(pts_idx, dim=0)
        pts_idx = torch.clamp(pts_idx, 0, grid_size - 1)
        pts_list.append(pts_idx)
        # occs[b, pts_idx[:, 0], pts_idx[:, 1], pts_idx[:, 2]] = 1
    pts_masked = torch.stack(pts_list, dim=0)
    # return occs, pts_masked
    return pts_masked


def create_occ_gts(item_idx, device='cuda'):
    try:
        name = objaverse_key[item_idx]
        img_folder = os.path.join(data_root, name)
        
        c2ws, Ks, depths = [], [], []
        for index_cond in range(40):
            # Read camera info
            camera_path = os.path.join(
                img_folder, '{:05d}/{:05d}.json'.format(index_cond, index_cond))
            cond_c2w_org, fov_x, fov_y = read_camera_info_single(camera_path)
            transform_matrix = np.diag([1, -1, -1, 1])  # opencv to blender
            cond_c2w = cond_c2w_org @ transform_matrix
            cond_c2w = torch.from_numpy(cond_c2w).float()
            cond_cam_dis = np.linalg.norm(cond_c2w[:3, 3:], 2)
            K = FOV_to_intrinsics(fov_x, degree=False).float()
            # Note: K is normalized by image size
            c2ws.append(cond_c2w)
            Ks.append(K)
            
            # Read normal and depth
            normald_path = os.path.join(
                img_folder, '{:05d}/{:05d}_nd.exr'.format(index_cond, index_cond))
            normald = cv2.imread(
                normald_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            
            depth = normald[..., -1:]
            near_distance = cond_cam_dis - 0.867  # near, sqrt(3) * 0.5
            depth_mask = depth < near_distance
            kernel = np.ones((3, 3), dtype=np.uint8)
            depth_mask = cv2.dilate(depth_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            # We note that the edge of the depth map is not reliable
            depth[depth_mask] = 0
            depth = torch.from_numpy(depth).float()
            depths.append(depth)
        
        depths = torch.stack(depths, dim=0).to(device).permute(0, 3, 1, 2)
        c2ws = torch.stack(c2ws, dim=0).to(device)
        Ks = torch.stack(Ks, dim=0).to(device)
        
        # gt_occupancy, pts = get_occupancy(depths[None], c2ws[None], Ks[None])
        pts = get_occupancy(depths[None], c2ws[None], Ks[None])
        # occ_np = gt_occupancy[0].cpu().numpy().astype(bool)
        pts_np = pts[0].cpu().numpy()
        save_dir = os.path.join(occ_root, name)
        os.makedirs(save_dir, exist_ok=True)
        # np.save(os.path.join(save_dir, 'gt_occupancy.npy'), occ_np)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pts_np)
        # o3d.io.write_point_cloud(os.path.join(save_dir, 'pts.pcd'), pcd)
        np.save(os.path.join(save_dir, 'pts.npy'), pts_np.astype(np.uint8))
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=len(objaverse_key))
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    start, end = args.start, args.end
    
    os.makedirs(occ_root, exist_ok=True)
    # process_map(create_occ_gts, range(start, end), max_workers=args.num_workers, chunksize=1)
    for item_idx in tqdm(range(start, end)):
        create_occ_gts(item_idx)
    print('Done.')
