import cv2
import math
import numpy as np
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.transforms import augment
from PIL import Image
from . import utils
import random
import torch
from torch.nn import functional as F
from basicsr.data.degradations import (
    circular_lowpass_kernel, random_mixed_kernels,
    random_add_gaussian_noise_pt, random_add_poisson_noise_pt
)
from basicsr.data.transforms import augment
from basicsr.utils import img2tensor, DiffJPEG,get_root_logger
from basicsr.utils.img_process_util import filter2D
from cldm.model import instantiate_from_config
import time


class MixDegradationDataset(data.Dataset):
    
    def __init__(self, opt):
        super(MixDegradationDataset, self).__init__()
        
        self.opt = opt
        self.type_list = opt['type_list']
        self.type_prob = opt['type_prob']
        self.opt_gtpaired = opt['gtpaired_config']
        self.opt_realesrgan = opt['realesrgan_config']

        self.meta_info = opt['meta_info']
        self.paths = utils.parse_meta_info(self.meta_info)
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch, taken from Real-ESRGAN:
        https://github.com/xinntao/Real-ESRGAN

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.data.params.train.params.get('queue_size', b*50)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w)
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w)
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    def __getitem__(self, index):
        degradation_type =  random.choices(self.type_list, self.type_prob)[0]
        if degradation_type == 'realesrgan':
            return self.__getitem_realesrgan__(index)
        elif degradation_type == 'gtpaired':
            return self.__getitem_gtpaired__(index)
        else:
            print("We don't find your dataset!")

    @torch.no_grad()
    def __getitem_realesrgan__(self, index):
        self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts
        self.queue_size = self.opt_realesrgan.get('queue_size', 180)

        self.lq: torch.Tensor = None
        self.hq: torch.Tensor = None

        # a final sinc filter
        self.final_sinc_prob = self.opt_realesrgan['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        data = self.get_kernels(index)
        # training data synthesis
        self.gt = data['gt'][None, ...]
        gt_path = data['gt_path']
        
        self.kernel1 = data['kernel1'][None, ...]
        self.kernel2 = data['kernel2'][None, ...]
        self.sinc_kernel = data['sinc_kernel'][None, ...]

        ori_h, ori_w = self.gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(self.gt, self.kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt_realesrgan['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt_realesrgan['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt_realesrgan['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.opt_realesrgan['gray_noise_prob']
        if np.random.uniform() < self.opt_realesrgan['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt_realesrgan['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt_realesrgan['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt_realesrgan['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt_realesrgan['second_blur_prob']:
            out = filter2D(out, self.kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt_realesrgan['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt_realesrgan['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt_realesrgan['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / self.opt_realesrgan['scale'] * scale), int(ori_w / self.opt_realesrgan['scale'] * scale)), mode=mode)
        # add noise
        gray_noise_prob = self.opt_realesrgan['gray_noise_prob2']
        if np.random.uniform() < self.opt_realesrgan['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt_realesrgan['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt_realesrgan['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt_realesrgan['scale'], ori_w // self.opt_realesrgan['scale']), mode=mode)
            out = filter2D(out, self.sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt_realesrgan['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt_realesrgan['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.opt_realesrgan['scale'], ori_w // self.opt_realesrgan['scale']), mode=mode)
            out = filter2D(out, self.sinc_kernel)
        
        # resize back to size of ground truth
        if self.opt_realesrgan['scale'] != 1:
            out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic", antialias=True)
        # clamp and round
        self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        
        self._dequeue_and_enqueue()

        # self.gt: rgb, nchw, 0,1
        # hq: rgb, hwc, -1,1
        hq = (self.gt[0].permute(1, 2, 0) * 2 - 1).contiguous().numpy().astype(np.float32)
        # self.lq: rgb, nchw, 0,1
        # lq: rgb, hwc, 0,1
        lq = (self.lq[0].permute(1, 2, 0) * 2 - 1).contiguous().numpy().astype(np.float32)
        return dict(jpg=hq, txt="", hint=lq, gt_path=gt_path, type="realesrgan" )

    def get_kernels(self, index):
        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # hwc, rgb, 0,255
        retry = 3
        while retry > 0:
            try:
                if self.opt_realesrgan['use_crop']:
                    if self.opt_realesrgan['crop_type'] == 'center_crop':
                        pil_img_gt = utils.center_crop_arr(Image.open(gt_path).convert("RGB"), self.opt_realesrgan['out_size'])
                    elif self.opt_realesrgan['crop_type'] == 'random_crop':
                        pil_img_gt = utils.random_crop_arr(Image.open(gt_path).convert("RGB"), self.opt_realesrgan['out_size'])
                    else:
                        raise ValueError(self.opt_realesrgan['crop_type'])
                else:
                    pil_img_gt = np.array(Image.open(gt_path).convert("RGB"))
                    assert pil_img_gt.shape[:2] == (self.opt_realesrgan['out_size'], self.opt_realesrgan['out_size'])
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'dataloader error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(30)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        
        # hwc, bgr, 0,1 (same as the original code)
        img_gt = (pil_img_gt[..., ::-1].copy() / 255.0).astype(np.float32)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt_realesrgan['use_hflip'], self.opt_realesrgan['use_rot'])

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt_realesrgan['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.opt_realesrgan['kernel_list'],
                self.opt_realesrgan['kernel_prob'],
                kernel_size,
                self.opt_realesrgan['blur_sigma'],
                self.opt_realesrgan['blur_sigma'], [-math.pi, math.pi],
                self.opt_realesrgan['betag_range'],
                self.opt_realesrgan['betag_range'],
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt_realesrgan['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.opt_realesrgan['kernel_list2'],
                self.opt_realesrgan['kernel_prob2'],
                kernel_size,
                self.opt_realesrgan['blur_sigma2'],
                self.opt_realesrgan['blur_sigma2'], [-math.pi, math.pi],
                self.opt_realesrgan['betag_range2'],
                self.opt_realesrgan['betag_range2'],
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt_realesrgan['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d

    def __getitem_gtpaired__(self, index):

        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]

        # hwc, rgb, 0,255
        retry = 3
        while retry > 0:
            try:
                if self.opt_gtpaired['use_crop']:
                    pil_img_gt = utils.center_crop_arr(Image.open(gt_path).convert("RGB"), self.opt_gtpaired['out_size'])
                else:
                    pil_img_gt = np.array(Image.open(gt_path).convert("RGB"))
                    assert pil_img_gt.shape[:2] == (self.opt_gtpaired['out_size'], self.opt_gtpaired['out_size'])
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'dataloader error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(30)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        img_gt = (pil_img_gt[..., ::-1].copy() / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt_gtpaired['use_hflip'], rotation=False, return_status=True)
        h, w, _ = img_gt.shape

        # BGR to RGB, [-1, 1]
        target = ((img_gt[..., ::-1] - 0.5) / 0.5).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = ((img_gt[..., ::-1] - 0.5) / 0.5).astype(np.float32)

        return dict(jpg=target, txt="", hint=source, gt_path=gt_path,type="gtpaired") # gt_path is used in test_guidance.py

    def __len__(self):
        return len(self.paths)
