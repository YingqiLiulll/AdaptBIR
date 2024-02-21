import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from ldm.util import instantiate_from_config
import torch.nn.functional as F
import math
import os

def pad(img, scale):
    h, w = img.shape[:2]
    ph = 0 if h % scale == 0 else math.ceil(h / scale) * scale - h
    pw = 0 if w % scale == 0 else math.ceil(w / scale) * scale - w
    return np.pad(
        img, pad_width=((0, ph), (0, pw), (0, 0)), mode="constant",
        constant_values=0
    )

def make_batch(img, device):
    img = Image.open(img).convert("RGB")
    print("img.shape",img.size)
    ori_size = (img.size[-2],img.size[-1])
    s = min(img.size)
    if s < 512:
        r = 512 / s
        img = img.resize(
            tuple(math.ceil(x * r) for x in img.size), Image.LANCZOS
        )
    image = np.array(img)
    h, w, _ = image.shape
    image = pad(image,64)
    image = torch.tensor(image).permute(2, 0, 1)
    image = (image.float() / 127.5) - 1.0
    image = image.unsqueeze(0)
    image = image.to(device=device)
    print(image.shape)
    return image, h, w,ori_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help=
        "dir containing image-mask (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
		"--resize_back",
		action="store_true",
	)
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="?",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
    )
    opt = parser.parse_args()
    images = []
    for root, dirs, files in os.walk(opt.indir):
        for file in files:
            image_path = os.path.join(root, file)
            images.append(image_path)
    print(len(images))

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(opt.ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    os.makedirs(opt.outdir, exist_ok=True)

    i = 0
    with torch.no_grad():
        for image in tqdm(images):
            filepath, filename = os.path.split(image)
            name, suffix = os.path.splitext(filename)
            batch, h, w, ori_size = make_batch(image, device=device)
            predicted_image, _ = model(batch)
            batch = batch[:, :, :h, :w]
            predicted_image = predicted_image[:, :, :h, :w]
            predicted_image = predicted_image.clamp(-1, 1)
            print("Process image shape:",predicted_image.shape, "Orignal image shape:", batch.shape)
            predicted_image = (
                predicted_image.cpu().numpy().transpose(0, 2, 3, 1)[0])
            x = predicted_image
            x = ((x - x.min()) / (x.max() - x.min()) * 255).clip(0, 255).astype(np.uint8)
            i += 1
            opath = os.path.join(opt.outdir, "{}{}".format(name,suffix))
            def resize_back(img):
                return np.array(Image.fromarray(img).resize(ori_size, resample=Image.LANCZOS))
            
            if (w,h) != ori_size and opt.resize_back:
                 x = resize_back(x)

            Image.fromarray(x).save(opath)