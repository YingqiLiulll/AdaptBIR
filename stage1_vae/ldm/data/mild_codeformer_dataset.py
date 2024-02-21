import cv2
import math
import numpy as np
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.transforms import augment

from PIL import Image
from . import utils
import random


class DegradationDataset(data.Dataset):
    
    def __init__(self, opt):
        super(DegradationDataset, self).__init__()
        
        self.opt = opt
        self.meta_info = self.opt['meta_info']
        self.out_size = opt['out_size']
        self.use_crop = opt['use_crop']
        self.paths = utils.parse_meta_info(self.meta_info)
        
        print("images: ")
        for i in range(min(10, len(self.paths))):
            print(f"\t{self.paths[i]}")
        print(f"\t... size = {len(self.paths)}")
        print(f"use crop: {self.use_crop}")

        # degradation configurations
        self.blur_probability = opt['blur_probability']
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']

    def __getitem__(self, index):
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        if self.use_crop:
            pil_img_gt = utils.center_crop_arr(Image.open(gt_path).convert("RGB"), self.out_size)
        else:
            pil_img_gt = np.array(Image.open(gt_path).convert("RGB"))
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        img_gt = (pil_img_gt[..., ::-1].copy() / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        h, w, _ = img_gt.shape

        # ------------------------ generate lq image ------------------------ #
        # blur
        if round(np.random.uniform(0, 1), 1) <= self.blur_probability:
            if not isinstance(self.blur_kernel_size, int):
                blur_kernel_list = list(range(self.blur_kernel_size[0], self.blur_kernel_size[1], self.blur_kernel_size[2]))
                self.blur_kernel_size = random.choice(blur_kernel_list)
                print("self.blur_kernel_size:",self.blur_kernel_size)
            kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                noise_range=None)
            img_lq = cv2.filter2D(img_gt, -1, kernel)
        else:
            img_lq = img_gt
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [-1, 1]
        target = ((img_gt[..., ::-1] - 0.5) / 0.5).astype(np.float32)
        # BGR to RGB, [-1, 1]
        source = ((img_lq[..., ::-1] - 0.5) / 0.5).astype(np.float32)

        return dict(image=source, txt="", target=target, gt_path=gt_path) # gt_path is used in test_guidance.py

    def __len__(self):
        return len(self.paths)
