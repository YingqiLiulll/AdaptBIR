import os
import pyiqa
import torch
from tqdm import tqdm
import numpy as np
import time
import cv2
from argparse import ArgumentParser
import piq

# from bfrxlib.utils.registry import METRIC_REGISTRY

musiq_model = None
# @METRIC_REGISTRY.register()
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
# @METRIC_REGISTRY.register()
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
    txt_file = open('/cpfs01/user/hejingwen/projects/DiffBIR_private-main/stage2/DIV2k_valid_musiq_maniqa.txt','w')
    parser = ArgumentParser()
    parser.add_argument("--img_folder", required=True, type=str)

    # ----------------- args ----------------- #
    args = parser.parse_args()
    files = find_images(args.img_folder)
    list_musiq = []
    list_maniqa = []

    for file in files:
        img_name = file
        img = cv2.imread(file, flags=1)
        musiq = calculate_musiq(img)
        maniqa = calculate_maniqa(img)
        print("img_name:",img_name,file=txt_file,end='\n')
        print("musiq:",musiq,file=txt_file,end='\n')
        print("maniqa:",maniqa,file=txt_file,end='\n')

        # print("niqe:",niqe)
        list_musiq.append(musiq)
        list_maniqa.append(maniqa)

    print("len(list_niqe):",len(list_musiq))
    print("平均musiq:", np.mean(list_musiq))  # list_musiq
    print("平均maniqa:", np.mean(list_maniqa))

if __name__ == "__main__":
    main()