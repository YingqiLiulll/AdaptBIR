import os
import pyiqa
import torch
from tqdm import tqdm
import numpy as np
import time
import cv2
from argparse import ArgumentParser
import piq
import math

musiq_model = None
def calculate_musiq(data, device=None, **kargs):
    """
    data: {'restored_img': img # BGR, HWC, [0, 255]}
    """
    # img = data['img_restored']
    img = data
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    global musiq_model
    if musiq_model is None:
        # TODO: to load model only once can run 10 times faster
        musiq_model = pyiqa.create_metric('musiq', device=device, as_loss=False)
    
    # (N, C, H, W), RGB, [0, 1]
    img = np.array(img[..., ::-1]).transpose(2, 0, 1) / 255
    img = torch.tensor(img[None, ...]).float().to(device)
    # calculate score
    score = musiq_model(img) # torch.Tensor, size = (1, 1)
    # score = float(score.detach().cpu().numpy()[0][0])
    musiq_score = float(score.detach().cpu().numpy())

    return musiq_score

maniqa_model = None
def calculate_maniqa(data, device=None, **kargs):
    """
    data: {'restored_img': img # BGR, HWC, [0, 255]}
    """
    # img = data['img_restored']
    img = data
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    global maniqa_model
    if maniqa_model is None:
        maniqa_model = pyiqa.create_metric('maniqa', device=device, as_loss=False)
    # metric = pyiqa.create_metric('maniqa', device=device, as_loss=False)
    
    img = img[..., ::-1].copy() # RGB, HWC, [0, 255]
    
    # (N, C, H, W), RGB, [0, 1]
    img = np.array(img).transpose(2, 0, 1) / 255
    img = torch.tensor(img[None, ...]).float().to(device)
    # calculate score
    # print(type(score)) # torch.Tensor, size = (1, 1)
    maniqa_score = float(maniqa_model(img).detach().cpu().numpy()[0][0])

    return maniqa_score

def find_images(imgdir):
    res = []
    for dirpath, _, filenames in os.walk(imgdir):
        for filename in filenames:
            if filename.split(".")[-1].lower() in ["jpg", "png", "jpeg"]:
                res.append(os.path.abspath(os.path.join(dirpath, filename)))
    return res

def main():
    txt_file1 = open('/AdaptBIR/Valid/class/CommonIR/medium/mild_class.txt','w')
    txt_file2 = open('/AdaptBIR/Valid/class/CommonIR/medium/medium_class.txt','w')
    txt_file3 = open('/AdaptBIR/Valid/class/CommonIR/medium/severe_class.txt','w')
    parser = ArgumentParser()
    parser.add_argument("--img_folder", required=True, type=str)
    parser.add_argument("--gt_folder",required=True,type=str)

    # ----------------- args ----------------- #
    args = parser.parse_args()
    files = find_images(args.img_folder)
    list_mild_psnr = []
    list_medium_psnr = []
    list_severe_psnr = []
    list_mild_SSIM = []
    list_medium_SSIM = []
    list_severe_SSIM = []
    list_mild_LPIPS = []
    list_medium_LPIPS = []
    list_severe_LPIPS = []
    # list_maniqa = []

    for file in files:
        img_name = file
        img = cv2.imread(file, flags=1)
        h,w,c = img.shape
        l = min(h,w)
        if l < 512:
            r = 512 / l
            img_upsample = cv2.resize(img,(math.ceil(h * r), math.ceil(w * r)))
        else:
            img_upsample = img
        musiq = calculate_musiq(img_upsample)
        # maniqa = calculate_maniqa(img)
        gt = cv2.imread(os.path.join(args.gt_folder, file.split("/")[-1]),flags=1)

        img = img[:,:,::-1]
        gt = gt[:,:,::-1]
        img_a = np.array(img,dtype="float32")
        img_b = np.array(gt,dtype="float32")
        img_a = torch.Tensor(img_a).permute(2,0,1).unsqueeze(0).cuda()
        img_b = torch.Tensor(img_b).permute(2,0,1).unsqueeze(0).cuda()

        psnr_num = piq.psnr(img_a, img_b,data_range=255.)
        ssim_num = piq.ssim(img_a, img_b,data_range=255.)
        # mse_num = mse(img_a, img_b)
        img_a = img_a/255.
        img_b = img_b/255.
        lpips_func = piq.LPIPS(replace_pooling=True).cuda()
        lpips_num = lpips_func(img_a , img_b)
        assert musiq >=0
        if musiq <= 45:
            print("img_name:",img_name,file=txt_file3,end='\n')
            list_severe_psnr.append(psnr_num)
            list_severe_SSIM.append(ssim_num)
            list_severe_LPIPS.append(lpips_num)
        elif 45 < musiq <= 65:
            print("img_name:",img_name,file=txt_file2,end='\n')
            list_medium_psnr.append(psnr_num)
            list_medium_SSIM.append(ssim_num)
            list_medium_LPIPS.append(lpips_num)
        else:
            print("img_name:",img_name,file=txt_file1,end='\n')
            list_mild_psnr.append(psnr_num)
            list_mild_SSIM.append(ssim_num)
            list_mild_LPIPS.append(lpips_num)

        print(f"PSNR: {psnr_num}, SSIM: {ssim_num}, LPIPS: {lpips_num}")

    if list_mild_psnr:
        print("mild平均PSNR:", torch.stack(list_mild_psnr,dim=0).mean(dim=0))  
        print("mild平均SSIM:", torch.stack(list_mild_SSIM,dim=0).mean(dim=0))  
        print("mild平均LPIPS:", torch.stack(list_mild_LPIPS,dim=0).mean(dim=0))  

    if list_medium_psnr:
        print("medium平均PSNR:", torch.stack(list_medium_psnr,dim=0).mean(dim=0))  
        print("medium平均SSIM:", torch.stack(list_medium_SSIM,dim=0).mean(dim=0))  
        print("medium平均LPIPS:", torch.stack(list_medium_LPIPS,dim=0).mean(dim=0))  

    if list_severe_psnr:
        print("severe平均PSNR:", torch.stack(list_severe_psnr,dim=0).mean(dim=0))  
        print("severe平均SSIM:", torch.stack(list_severe_SSIM,dim=0).mean(dim=0))  
        print("severe平均LPIPS:", torch.stack(list_severe_LPIPS,dim=0).mean(dim=0)) 

if __name__ == "__main__":
    main()