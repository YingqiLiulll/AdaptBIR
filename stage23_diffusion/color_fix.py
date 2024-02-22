from PIL import Image
import os
from PIL import ImageFile
import math
from cldm.align_color import wavelet_reconstruction, adaptive_instance_normalization
import numpy as np
import einops
import torch 

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

scale = 4
auto_mode_size=512

# This is a script for directly processing color fix.

color_fix_type = 'wavelet'

sr_path = "path to Images waiting to be processed"
gt_path = 'path to GT'
output_path = 'path to output'
f = os.listdir(gt_path)
for i in f:
    print("picture:",i)
    pf_a = os.path.join(gt_path, i)
    gt = Image.open(pf_a).convert('RGB')
    img_name,suffix = i.split(".")
    pf_b = os.path.join(sr_path, img_name+'_0.'+ suffix)
    sr = Image.open(pf_b).convert('RGB')

    sr = ((np.array(sr)/ 255.0)*2.0 -1.0).astype(np.float32)
    gt = ((np.array(gt)/ 255.0)*2.0 -1.0).astype(np.float32)

    control_img = (torch.tensor(sr)).unsqueeze(0)
    gt_img = (torch.tensor(gt)).unsqueeze(0)

    if color_fix_type == 'adain':
        print("apply adain color correction")
        x_samples = adaptive_instance_normalization(control_img, gt_img)
    elif color_fix_type == 'wavelet':
        print("apply wavelet color correction")
        x_samples = wavelet_reconstruction(control_img, gt_img)
    else:
        print("apply none color correction")
        assert color_fix_type == "none", f"unexpected color fix type: {color_fix_type}"

    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    results = x_samples[0]


    pf_output = os.path.join(output_path, i)
    Image.fromarray(results).save(pf_output)

