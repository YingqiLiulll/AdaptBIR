import cv2
import math
import numpy as np
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.transforms import augment

from PIL import Image
from . import utils


class GtpairDataset(data.Dataset):
    
    def __init__(self, opt):
        super(GtpairDataset, self).__init__()
        
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
        
        # BGR to RGB, [-1, 1]
        target = ((img_gt[..., ::-1] - 0.5) / 0.5).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = ((img_gt[..., ::-1] - 0.5) / 0.5).astype(np.float32)

        return dict(jpg=target, txt="", hint=source, gt_path=gt_path) # gt_path is used in test_guidance.py

    def __len__(self):
        return len(self.paths)
