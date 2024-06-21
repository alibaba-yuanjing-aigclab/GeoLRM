# Copyright (C) 2024-present Yuanjing Shengsheng (Beijing) Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details].

import os, sys
import json
from pathlib import Path
from functools import partial

import cv2
import random
import albumentations
import numpy as np
from PIL import Image
import webdataset as wds
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as TF
from einops import rearrange

file_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.join(file_dir, '..', '..')
sys.path.append(project_root)  # for debugging
from src.datasets.image_degradation import (
    degradation_fn_bsr,
    degradation_fn_bsr_light)
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    center_looking_at_camera_pose, 
    get_circular_camera_poses,
)

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


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


def unity2blender(normal):
    normal_clone = normal.copy()
    normal_clone[..., 0] = -normal[..., 2]
    normal_clone[..., 1] = -normal[..., 0]
    normal_clone[..., 2] = normal[..., 1]

    return normal_clone


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=1, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class GObjaverseDatasetOcc(Dataset):
    def __init__(self,
                 json_path=None,
                 caption_path=None,
                 data_root=None,
                 occ_data_root='data/objaverse/gobjaverse_280k_occ',
                 size=512,
                 degradation=None,
                 downscale_f=1,
                 min_crop_f=0.8,
                 max_crop_f=1.,
                 random_crop=False,
                 debug: bool = False,
                 orth_views=24,
                 validation=False,
                 folder_key='',
                 as_video=True,
                 pre_str='',
                 suff_str=', 3d asset',
                 albedo_check=False,
                 filter_box: bool = True,
                #  color_key='_albedo',
                 select_view=[0, 6, 12, 18],
                 rand_target=False,
                 rand_views=40,
                 fix_view=False):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        :param as_video: as video containers?
        :param pre_str: the pre string of caption?
        :param views: rendering views

        """
        assert json_path is not None
        self.items = self.read_json(json_path)
        self.data_root = data_root
        self.occ_data_root = occ_data_root
        # self.color_key = color_key
        self.objaverse_key = sorted(self.items)

        with open(caption_path, 'r') as reader:
            self.folder2caption = json.load(reader)

        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert (max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(
            max_size=size, interpolation=cv2.INTER_AREA)
        self.pil_interpolation = False  # gets reset later if incase interp_op is from pillow
        self.orth_views = orth_views
        self.validation = validation
        self.folder_key = folder_key

        self.debug = debug
        self.as_video = as_video
        self.pre_str = pre_str
        self.suff_str = suff_str
        self.albedo_check = albedo_check
        self.filter_box = filter_box
        self.select_view = select_view
        self.rand_target = rand_target
        self.rand_views = rand_views
        self.fix_view = fix_view

        if degradation == 'bsrgan':
            self.degradation_process = partial(
                degradation_fn_bsr, sf=downscale_f)

        elif degradation == 'bsrgan_light':
            self.degradation_process = partial(
                degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fns = {
                'cv_nearest': cv2.INTER_NEAREST,
                'cv_bilinear': cv2.INTER_LINEAR,
                'cv_bicubic': cv2.INTER_CUBIC,
                'cv_area': cv2.INTER_AREA,
                'cv_lanczos': cv2.INTER_LANCZOS4,
                'pil_nearest': Image.NEAREST,
                'pil_bilinear': Image.BILINEAR,
                'pil_bicubic': Image.BICUBIC,
                'pil_box': Image.BOX,
                'pil_hamming': Image.HAMMING,
                'pil_lanczos': Image.LANCZOS,
            }
            interpolation_fn = interpolation_fns[degradation]

            self.pil_interpolation = degradation.startswith('pil_')

            if self.pil_interpolation:
                self.degradation_process = partial(
                    TF.resize,
                    size=self.LR_size,
                    interpolation=interpolation_fn)
            else:
                self.degradation_process = albumentations.SmallestMaxSize(
                    max_size=self.LR_size, interpolation=interpolation_fn)
        
        print(f'length of dataset: {len(self)}')

    def read_json(self, json_file):
        with open(json_file, 'r') as reader:
            return json.load(reader)

    def __len__(self):
        # for validation only 100 items
        return len(self.objaverse_key) if not self.validation else 10

    def valid(self, data):
        shape_valid = data['normals'].shape[1] == 3 and \
            data['depths'].shape[1] == 1 and \
            data['images'].shape[1] == 3
        num_valid = self.isnot_nan(data)
        return shape_valid and num_valid

    def retrival(self, name, index_cond, normal_bg=[0, 0, 1], image_bg=[1, 1, 1]):
        """Retrival the data from the folder
        
        Returns:
            image: [H, W, 3]
            alpha: [H, W, 1]
            normal: [H, W, 3]
            depth: [H, W, 1]
            cond_c2w: [4, 4]
            K: [3, 3]
        """
        img_folder = os.path.join(self.data_root, name, self.folder_key)
        
        # Read image and alpha
        img_path = os.path.join(
            img_folder, '{:05d}/{:05d}.png'.format(index_cond, index_cond))
        pil_img = Image.open(img_path)
        image = np.asarray(pil_img, dtype=np.float32) / 255.
        image = self.image_rescaler(image=image)['image']
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + image_bg * (1 - alpha)
        image = torch.from_numpy(image).float()
        alpha = torch.from_numpy(alpha).float()
        
        # Read camera info
        camera_path = os.path.join(
            img_folder, '{:05d}/{:05d}.json'.format(index_cond, index_cond))
        cond_c2w_org, fov_x, fov_y = read_camera_info_single(camera_path)
        transform_matrix = np.diag([1, -1, -1, 1])  # opencv to blender
        cond_c2w = cond_c2w_org @ transform_matrix
        cond_cam_dis = np.linalg.norm(cond_c2w[:3, 3:], 2)
        cond_c2w = torch.from_numpy(cond_c2w).float()
        K = FOV_to_intrinsics(fov_x, degree=False).float()
        # Note: K is normalized by image size
        
        # Read normal and depth
        normald_path = os.path.join(
            img_folder, '{:05d}/{:05d}_nd.exr'.format(index_cond, index_cond))
        normald = cv2.imread(
            normald_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        normald = self.image_rescaler(image=normald)['image']
        
        depth = normald[..., -1:]
        near_distance = cond_cam_dis - 0.867  # near, sqrt(3) * 0.5
        depth_mask = depth < near_distance
        kernel = np.ones((3, 3), dtype=np.uint8)
        depth_mask = cv2.dilate(depth_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        # We note that the edge of the depth map is not reliable
        depth[depth_mask] = 0
        depth = torch.from_numpy(depth).float()
        
        normal = unity2blender(normald[..., :3])
        normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = np.nan_to_num(normal, nan=-1.0)
        normal = torch.from_numpy(normal).float()
        normal = (normal + 1) / 2
        normal = torch.lerp(torch.zeros_like(normal), normal, alpha)

        return image, alpha, normal, depth, cond_c2w, K

    def debug_img(self, img, path='debug', name='example.png'):

        img = rearrange(img, 'b c h w -> h (b w) c')

        img = (img + 1) / 2
        img = (img.detach().cpu() * 255).numpy().astype(np.uint8)

        save_name = os.path.join(path, name)

        if img.shape[-1] == 1:
            img = img[..., 0]
        else:
            img = img[..., [2, 1, 0]]

        cv2.imwrite(save_name, img)

    def contain_views(self, objaverse_name, view_ids):

        image_list = []
        alpha_list = []
        normal_list = []
        depth_list = []
        # albedo_list = []
        c2w_list = []
        K_list = []

        for view in view_ids:
            image, normal_mask, normal, depth, cond_c2w, K = self.retrival(
                objaverse_name, view)
            image_list.append(image)
            alpha_list.append(normal_mask)
            normal_list.append(normal)
            depth_list.append(depth)
            # albedo_list.append(albedo)
            c2w_list.append(cond_c2w)
            K_list.append(K)

        format_list = lambda x: torch.stack(x, dim=0).permute(0, 3, 1, 2)
        ret_dict = {
            "images": format_list(image_list),
            "alphas": format_list(alpha_list),
            "normals": format_list(normal_list),
            "depths": format_list(depth_list),
            # "albedo": format_list(albedo_list),
            "c2ws": torch.stack(c2w_list, dim=0),
            "Ks": torch.stack(K_list, dim=0),  # Note: normalized by image size
        }
        return ret_dict

    def isnot_nan(self, ret_dict):
        normal = ret_dict['normals']
        depth = ret_dict['depths']
        # albedo = ret_dict['albedo']

        return torch.isnan(normal).sum() == 0 and torch.isnan(depth).sum() == 0
        # and torch.isnan(albedo).sum() == 0

    def __getitem__(self, item):

        ret_dict = {}

        while True:
            try:
                objaverse_name = self.objaverse_key[item]
                caption = self.folder2caption[objaverse_name]

                if self.filter_box:
                    if 'box' in caption or 'cube' in caption:
                        raise ValueError('containing cube or box in caption')

                # TODO: (p2) pose rotation
                index_cond = random.sample(range(self.orth_views), 1)[0] if not self.fix_view else 0
                select_views = [(i + index_cond) % self.orth_views for i in self.select_view]
                input_dict = self.contain_views(objaverse_name, select_views)
                
                if self.rand_target:
                    options = list(set(range(self.rand_views)) - set(select_views))
                    target_views = random.sample(options, len(select_views))
                else:
                    target_views = [(i + 3) % self.orth_views for i in self.select_view]  # 45 degree
                
                target_dict = self.contain_views(objaverse_name, target_views)

                assert (self.valid(input_dict) and self.valid(target_dict))
                
                for key in input_dict:
                    ret_dict['input_' + key] = input_dict[key]
                    ret_dict['target_' + key] = target_dict[key]

                # if self.albedo_check:
                #     # all white about 10 % data is all white
                #     assert not (albedo == 1).sum() / albedo.numel() > 0.95

                # ret_dict['caption'] = self.pre_str + caption[:-1] + self.suff_str
                pts_path = os.path.join(self.occ_data_root, objaverse_name, 'pts.npy')
                pts = np.load(pts_path)
                pts = torch.from_numpy(pts).long()
                occupancy = torch.zeros(128, 128, 128, dtype=torch.float32)
                occupancy[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
                ret_dict['occupancy'] = occupancy
                
                if self.debug:
                    self.debug_img(
                        input_dict["images"], name='images_{:04d}.png'.format(item))
                    self.debug_img(
                        input_dict["normals"], name='normals_{:04d}.png'.format(item))
                    self.debug_img(
                        input_dict["depths"], name='depths_{:04d}.png'.format(item))
                break

            except:
                item = (item + 1) % len(self)

        return ret_dict

    # using to debug
    def visualize(self, img, normal, depth):

        def to_01(img):
            return np.clip((img + 1) / 2, 0., 1.)

        img = to_01(img)
        normal = to_01(normal)
        depth = to_01(depth)
        plt.imsave('./debug/vis_objaverse/image.png', img)
        plt.imsave('./debug/vis_objaverse/depth.png', depth)
        plt.imsave('./debug/vis_objaverse/normal.png', normal)


class ValidationData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        input_view_num=6,
        input_image_size=320,
        fov=30,
    ):
        self.root_dir = Path(root_dir)
        self.input_view_num = input_view_num
        self.input_image_size = input_image_size
        self.fov = fov

        self.paths = sorted(os.listdir(self.root_dir))
        print('============= length of dataset %d =============' % len(self.paths))

        cam_distance = 4.0
        azimuths = np.array([30, 90, 150, 210, 270, 330])
        elevations = np.array([20, -10, 20, -10, 20, -10])
        azimuths = np.deg2rad(azimuths)
        elevations = np.deg2rad(elevations)

        x = cam_distance * np.cos(elevations) * np.cos(azimuths)
        y = cam_distance * np.cos(elevations) * np.sin(azimuths)
        z = cam_distance * np.sin(elevations)

        cam_locations = np.stack([x, y, z], axis=-1)
        cam_locations = torch.from_numpy(cam_locations).float()
        c2ws = center_looking_at_camera_pose(cam_locations)
        self.c2ws = c2ws.float()
        self.Ks = FOV_to_intrinsics(self.fov).unsqueeze(0).repeat(6, 1, 1).float()

        render_c2ws = get_circular_camera_poses(M=8, radius=cam_distance, elevation=20.0)
        render_Ks = FOV_to_intrinsics(self.fov).unsqueeze(0).repeat(render_c2ws.shape[0], 1, 1)
        self.render_c2ws = render_c2ws.float()
        self.render_Ks = render_Ks.float()

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)
        pil_img = pil_img.resize((self.input_image_size, self.input_image_size), resample=Image.BICUBIC)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        if image.shape[-1] == 4:
            alpha = image[:, :, 3:]
            image = image[:, :, :3] * alpha + color * (1 - alpha)
        else:
            alpha = np.ones_like(image[:, :, :1])

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def __getitem__(self, index):
        # load data
        input_image_path = os.path.join(self.root_dir, self.paths[index])

        '''background color, default: white'''
        bkg_color = [1.0, 1.0, 1.0]

        image_list = []
        alpha_list = []

        for idx in range(self.input_view_num):
            image, alpha = self.load_im(os.path.join(input_image_path, f'{idx:03d}.png'), bkg_color)
            image_list.append(image)
            alpha_list.append(alpha)
        
        images = torch.stack(image_list, dim=0).float()
        alphas = torch.stack(alpha_list, dim=0).float()

        data = {
            'input_images': images,
            'input_alphas': alphas,
            'input_c2ws': self.c2ws,
            'input_Ks': self.Ks,

            'render_c2ws': self.render_c2ws,
            'render_Ks': self.render_Ks,
        }
        return data


if __name__ == '__main__':
    obj_dataset = GObjaverseDataset(
        json_path='data/objaverse/gobjaverse_280k.json',
        caption_path='data/objaverse/text_captions_cap3d.json',
        data_root='data/objaverse/gobjaverse_280k',
        degradation='cv_bilinear',
        orth_views=24,
        debug=True,
        pre_str='the albedo of ',
        folder_key='')

    print(f'length of dataset: {len(obj_dataset)}')
    for i in range(0, len(obj_dataset)):
        data = obj_dataset[i]
        import pdb; pdb.set_trace()
