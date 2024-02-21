import random
import numpy as np
from pathlib import Path
from einops import rearrange

import torch
import torchvision as thv
from torch.utils.data import Dataset

# from utils import util_sisr
# from utils import util_image
from . import util_image
# from utils import util_common

from basicsr.data.realesrgan_dataset import RealESRGANDataset

class BaseDataFolder(Dataset):
    def __init__(
            self,
            dir_path,
            dir_path_gt,
            need_gt_path=True,
            length=None,
            ext=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            mean=0.5,
            std=0.5,
            ):
        super(BaseDataFolder, self).__init__()
        if isinstance(ext, str):
            files_path = [str(x) for x in Path(dir_path).glob(f'*.{ext}')]
        else:
            assert isinstance(ext, list) or isinstance(ext, tuple)
            files_path = []
            for current_ext in ext:
                files_path.extend([str(x) for x in Path(dir_path).glob(f'*.{current_ext}')])
        self.files_path = files_path if length is None else files_path[:length]
        self.dir_path_gt = dir_path_gt
        self.need_gt_path = need_gt_path
        self.mean=mean
        self.std=std

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, index):
        im_path = self.files_path[index]
        im = util_image.imread(im_path, chn='rgb', dtype='float32')
        im = util_image.normalize_np(im, mean=self.mean, std=self.std, reverse=False)
        # im = rearrange(im, 'h w c -> c h w')
        out_dict = {'image':im.astype(np.float32), 'lq':im.astype(np.float32)}

        if self.need_gt_path:
            out_dict['path'] = im_path

        if self.dir_path_gt is not None:
            gt_path = str(Path(self.dir_path_gt) / Path(im_path).name)
            im_gt = util_image.imread(gt_path, chn='rgb', dtype='float32')
            im_gt = util_image.normalize_np(im_gt, mean=self.mean, std=self.std, reverse=False)
            # im_gt = rearrange(im_gt, 'h w c -> c h w')
            out_dict['gt'] = im_gt.astype(np.float32)

        return out_dict
