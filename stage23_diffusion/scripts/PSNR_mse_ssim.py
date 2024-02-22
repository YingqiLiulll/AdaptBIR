import os
import numpy as np
import math
from PIL import Image
import torch
import piq

import time
from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?

tis =time.perf_counter()

def psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 10 * math.log10(255.0 * 255.0 / mse)


def mse(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    return mse


def ssim(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def lpips(y_true, y_pred):
    image1_tensor = torch.tensor(y_true).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(y_pred).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    lpips = perceptual_loss(image1_tensor, image2_tensor)
    return lpips

path1 = '/cpfs01/user/hejingwen/projects/DiffBIR_private-main/stage2/results/SR_testsets_IQA_musiq/stage3_add_cfg1/LQ_codeformer_severe/'  # 指定输出结果文件夹
path2 = '/cpfs01/user/hejingwen/Datasets/SR_testsets/GT/'  # 指定原图文件夹
f = os.listdir(path2)
perceptual_loss = LPIPS().eval()
# f_nums = len(os.listdir(path1))
list_psnr = []
list_ssim = []
list_mse = []
list_lpips = []
for i in f:
    pf_a = os.path.join(path2, i)
    img_a = Image.open(pf_a).convert('RGB')
    img_name,suffix = i.split(".")
    # pf_b = os.path.join(path1, img_name+'.'+ suffix)
    pf_b = os.path.join(path1, img_name+'_0.'+ suffix)
    # pf_b = os.path.join(path1, img_name+'_out.'+suffix)
    img_b = Image.open(pf_b).convert('RGB')
    img_a = np.array(img_a,dtype="float32")
    img_b = np.array(img_b,dtype="float32")
    img_a = torch.Tensor(img_a).permute(2,0,1).unsqueeze(0).cuda()
    img_b = torch.Tensor(img_b).permute(2,0,1).unsqueeze(0).cuda()

    psnr_num = piq.psnr(img_a, img_b,data_range=255.)
    ssim_num = piq.ssim(img_a, img_b,data_range=255.)
    # mse_num = mse(img_a, img_b)
    img_a = img_a/255.
    img_b = img_b/255.
    lpips_func = piq.LPIPS(replace_pooling=True).cuda()
    lpips_num = lpips_func(img_a , img_b)
    list_ssim.append(ssim_num)
    list_psnr.append(psnr_num)
    # list_mse.append(mse_num)
    list_lpips.append(lpips_num)
    print(f"PSNR: {psnr_num}, SSIM: {ssim_num}, LPIPS: {lpips_num}")
print("平均PSNR:", torch.stack(list_psnr,dim=0).mean(dim=0)) # ,list_psnr)
print("平均SSIM:", torch.stack(list_ssim,dim=0).mean(dim=0)) # ,list_ssim)
# print("平均MSE:", np.mean(list_mse))  # ,list_mse)
print("平均LPIPS:", torch.stack(list_lpips,dim=0).mean(dim=0))  # ,list_mse)

print("Time used:", tis)
