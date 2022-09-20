import os
import cv2
import numpy as np
import random
import torch
import torch.utils.data as data
from tqdm import tqdm

from sritmo.util import to_pixel_samples, load_raw_hdr


class PatchHDRDataset(data.Dataset):
    def __init__(self, data_root, transform, cache = None, mode='train', patch_size = 128, patch_num = 32):
        super().__init__()
        self.transform = transform
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.mode = mode
        self.cache = cache

        self.data_root = os.path.join(data_root, mode)
        if mode == 'train':
            if isinstance(data_root, str):
                assert cache is None
                fnames = os.listdir(os.path.join(self.data_root, 'calib_hdr'))
            elif isinstance(data_root, list):
                assert cache == 'in_memory'
                fnames = data_root
        elif mode == 'val':
            fnames = os.listdir(os.path.join(self.data_root, 'calib_hdr'))
        else:
            raise NotImplementedError('Dataset mode [{}] not implemented'.format(mode))
        self.valid_list = []
        for name in tqdm(fnames):
            if '.exr' in name or '.hdr' in name:
                if cache is None:
                    self.valid_list.append(name)
                elif cache == 'in_memory':
                    self.valid_list.append(self._load_raw_hdr(name))
                else:
                    raise NotImplementedError("Cache type of [{}] is not supported.".format(cache))
        
        self.sph_coords = {
            1024: self._get_screen_img(1024, 512),
            2048: self._get_screen_img(2048, 1024),
            4096: self._get_screen_img(4096, 2048),
            4000: self._get_screen_img(4000, 2000),
        }

    def _get_screen_img(self, width, height):
        xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
        screen_points = np.stack([xx, yy], axis=-1)
        return (screen_points * 2 - 1) * np.array([np.pi, np.pi/2])

    def __getitem__(self, idx):
        name = self.valid_list[idx % len(self.valid_list)]
        full_hdr = load_raw_hdr(os.path.join(self.data_root, 'calib_hdr', name))
        full_ldr = cv2.imread(os.path.join(self.data_root, 'ldr', os.path.splitext(name)[0]+'.jpg')) / 255.
        full_ldr = full_ldr * 2 - 1
        full_hdr = np.log(full_hdr + 1e-6)
        # patch sampling
        H, W, _ = full_hdr.shape
        sph_coords = self.sph_coords.get(int(W))
        if sph_coords is None:
            sph_coords = self._get_screen_img(W, H)
            self.sph_coords[int(W)] = sph_coords
        spatial_scale = random.uniform(1, 4)
        # TODO: maybe resize image randomly from 2k to 8k
        w_lr = self.patch_size
        w_hr = round(w_lr * spatial_scale)
        x0 = random.randint(0, full_ldr.shape[-3] - w_hr)
        y0 = random.randint(0, full_ldr.shape[-2] - w_hr)
        crop_hr_ldr = full_ldr[x0: x0 + w_hr, y0: y0 + w_hr, :]
        crop_hr_hdr = full_hdr[x0: x0 + w_hr, y0: y0 + w_hr, :]
        crop_coords = sph_coords[x0: x0 + w_hr, y0: y0 + w_hr, :]
        crop_lr_ldr = cv2.resize(crop_hr_ldr, (w_lr, w_lr), interpolation=cv2.INTER_AREA)

        # augment
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5

        def augment(x):
            if hflip:
                x = np.flip(x, axis=-2)
            if vflip:
                x = np.flip(x, axis=-1)
            return x
        
        crop_lr_ldr = torch.from_numpy(augment(crop_lr_ldr).astype('float32').copy()).permute(2, 0, 1)
        crop_hr_ldr = torch.from_numpy(augment(crop_hr_ldr).astype('float32').copy()).permute(2, 0, 1)
        crop_hr_hdr = torch.from_numpy(augment(crop_hr_hdr).astype('float32').copy()).permute(2, 0, 1)
        crop_coords = torch.from_numpy(augment(crop_coords).astype('float32').copy()).permute(2, 0, 1)

        # sample on high resolution
        hr_coord, hr_hdr = to_pixel_samples(crop_hr_hdr)
        _, hr_ldr = to_pixel_samples(crop_hr_ldr)
        crop_coords = crop_coords.reshape(2, -1).permute(1, 0)

        sample_lst = np.random.choice(len(hr_coord), w_lr*w_lr, replace=False)
        hr_coord = hr_coord[sample_lst]
        hr_ldr = hr_ldr[sample_lst]
        hr_hdr = hr_hdr[sample_lst]
        glb_coord = crop_coords[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr_ldr.shape[-2]
        cell[:, 1] *= 2 / crop_hr_ldr.shape[-1]

        return {
            'lr_ldr': crop_lr_ldr,
            'local_coord': hr_coord,
            'global_coord': glb_coord,
            'cell': cell,
            'hr_ldr': hr_ldr,
            'hr_hdr': hr_hdr
        }

    def __len__(self):
        if self.mode == 'train':
            # repeat 20 times each epoch during training
            return len(self.valid_list) * 20
        else:
            return len(self.valid_list)

