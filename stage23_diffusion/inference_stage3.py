from typing import List, Tuple, Optional
import os
import math
import copy
from argparse import ArgumentParser
import torch.nn as nn
from collections import OrderedDict

import numpy as np
import torch
import einops
from pytorch_lightning import seed_everything
from PIL import Image
from omegaconf import OmegaConf
from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.util import log_txt_as_img, exists, instantiate_from_config

from cldm.share import *
from cldm.model import create_model, load_state_dict
from cldm.spaced_sampler import SpacedSampler
from cldm.color_print import cprint
from cldm.align_color import wavelet_reconstruction, adaptive_instance_normalization
import cldm.color_print as cp
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import pyiqa


warn = lambda s: cprint(s, fcolor="red", bgcolor="red", style="bold")

# copy from DifFace so as to load swinir model trained on DifFace codebase.
def reload_model(model, ckpt):
    if list(model.state_dict().keys())[0].startswith('module.'):
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = ckpt
        else:
            ckpt = OrderedDict({f'module.{key}':value for key, value in ckpt.items()})
    else:
        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = OrderedDict({key[7:]:value for key, value in ckpt.items()})
        else:
            ckpt = ckpt
    # model.load_state_dict(ckpt)
    # for gt_vae
    model.load_state_dict(ckpt, strict=False)

def process(model, control_imgs, height, width, steps, guess_mode, strength, scale,
            color_fix_type, disable_preprocess_model, delta, csft_w):
    n_samples = len(control_imgs)
    
    sampler = SpacedSampler(model, var_type="fixed_small")
    with torch.no_grad():
        H, W = height, width
        control = torch.tensor(np.stack(control_imgs, 0)).float().cuda()
        # control = torch.tensor(np.stack(control_imgs, 0)).float().cpu()
        control = einops.rearrange(control, 'n h w c -> n c h w').clone()
        
        if disable_preprocess_model:
            warn("disable preprocess model")
        else:
            warn("apply preprocess model on control")
            z1 = model.apply_denoise_encoder(control).mode()
            c_latent1 = model.first_stage_model.post_quant_conv(z1)

            h = model.first_stage_model.encoder(control)
            z2 = model.first_stage_model.quant_conv(h)
            posterior = DiagonalGaussianDistribution(z2).mode()
            c_latent2 = model.first_stage_model.post_quant_conv(posterior)
            c_latent = c_latent1*(1-delta) +c_latent2*delta
            control_output = model.first_stage_model.decoder(c_latent)
        
        cond = {"c_concat": [control_output], "c_crossattn": [model.get_learned_conditioning([""] * n_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], 
                   "c_crossattn": [model.get_learned_conditioning(["low quality, blurry, low-resolution, noisy, unsharp, weird textures"] * n_samples)]}
        
        warn("encode control to latent space as condition")
        cond["c_latent"] = [c_latent]

        batch_csft_w = torch.tensor([csft_w]).reshape(n_samples,1,1,1).cuda()
        cond["csft_w"] = [batch_csft_w]
        un_cond["c_latent"] = None if guess_mode else [c_latent]
        un_cond["csft_w"] = None if guess_mode else [batch_csft_w]
        
        shape = (4, H // 8, W // 8)
        x_T = torch.randn((n_samples, 4, H // 8, W // 8)).cuda().float()
        # x_T = torch.randn((n_samples, 4, H // 8, W // 8)).cpu().float()
        warn(f"latent shape = {shape}")
        
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, _ = sampler.sample(steps, n_samples, shape, cond,
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=un_cond,
                                    cond_fn=None, x_T=x_T)
        x_samples = model.decode_first_stage(samples)
        
        # apply color correction
        init_image = control_output
        if color_fix_type == 'adain':
            warn("apply adain color correction")
            x_samples = adaptive_instance_normalization(x_samples, init_image)
        elif color_fix_type == 'wavelet':
            warn("apply wavelet color correction")
            x_samples = wavelet_reconstruction(x_samples, init_image)
        else:
            warn("apply none color correction")
            assert color_fix_type == "none", f"unexpected color fix type: {color_fix_type}"
        
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(n_samples)]
    
    return results


def find_images(imgdir):
    res = []
    for dirpath, _, filenames in os.walk(imgdir):
        for filename in filenames:
            if filename.split(".")[-1].lower() in ["jpg", "png", "jpeg"]:
                res.append(os.path.abspath(os.path.join(dirpath, filename)))
    return res


def get_denoiseAE_preds(model, lq_imgs,delta):
    x = torch.tensor(np.stack(lq_imgs, 0)).float().cuda()
    # x = torch.tensor(np.stack(lq_imgs, 0)).float().cpu()
    x = einops.rearrange(x, 'n h w c -> n c h w').clone()
    z = model.apply_denoise_encoder(x).mode()
    c_latent1 = model.first_stage_model.post_quant_conv(z)

    z2 = model.first_stage_model.encode(x).mode()
    c_latent2 = model.first_stage_model.post_quant_conv(z2)

    c_latent = c_latent1*(1-delta) +c_latent2*delta

    control_output = model.first_stage_model.decoder(c_latent)
    control_output = (einops.rearrange(control_output, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().detach().numpy().clip(0, 255).astype(np.uint8)
    control_output = [control_output[i] for i in range(len(control_output))]
    return control_output


def pad(pil_image_array, scale=64):
    h, w = pil_image_array.shape[:2]
    ph, pw = 0, 0
    if h % scale != 0:
        ph = ((h // scale) + 1) * scale - h
    if w % scale != 0:
        pw = ((w // scale) + 1) * scale - w
    return np.pad(pil_image_array, pad_width=((0, ph), (0, pw), (0, 0)), mode="edge")


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
    img_copy = np.array(img).transpose(2, 0, 1) / 255
    img_copy = torch.tensor(img_copy[None, ...]).float().to(device)
    musiq_score = float(musiq_model(img_copy).detach().cpu().numpy())

    return musiq_score


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--config", required=True, type=str, help="training config")
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--reload_denoiseAE", action="store_true")
    parser.add_argument("--denoiseAE_ckpt", type=str, default="")
    
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--base_folder", type=str, required=True)
    
    parser.add_argument("--resize_mode", type=str, required=True, choices=["square", "auto", "none", "scale_auto"])
    parser.add_argument("--auto_mode_size", type=int, default=512)
    parser.add_argument("--auto_mode_scale", type=float, default=4)
    parser.add_argument("--repeat_times", type=int, default=1)
    
     # latent image guidance
    parser.add_argument("--use_guidance", action="store_true")
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_t_start", type=int, default=1001)
    parser.add_argument("--g_t_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=5)

    parser.add_argument("--disable_preprocess_model", action="store_true")
    parser.add_argument("--color_fix_type", type=str, default="none", choices=["wavelet", "adain", "none"])
    parser.add_argument("--resize_back", action="store_true")
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--skip_if_exist", action="store_true")
    
    # ----------------- args ----------------- #
    seed_everything(231)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    # ----------------- model ----------------- #
    model = create_model(cfg.model.config).cuda()
    # model = create_model(cfg.model.config).cpu()
    model.load_state_dict(load_state_dict(args.ckpt, location='cuda'), strict=False)

    if args.reload_denoiseAE:
        warn(f"reload denoiseAE model from {args.denoiseAE_ckpt}")
        assert hasattr(model, "preprocess_model")
        denoiseAE_ckpt = torch.load(args.denoiseAE_ckpt)
        if "state_dict" in denoiseAE_ckpt:
            denoiseAE_ckpt = denoiseAE_ckpt["state_dict"]
        reload_model(model.preprocess_model, denoiseAE_ckpt)

    if hasattr(model, "denoise_encoder"):
        warn(f"initialize denoise encoder for this model")
        model.init_denoise_encoder()
    model.eval()
    
    # ----------------- test loop ----------------- #
    warn(f"sampling {args.steps} steps using ddpm sampler")
    files = find_images(args.img_folder)
    os.makedirs(args.out_folder, exist_ok=True)
    
    for file in files:
        dirs = os.path.dirname(file).split("/")
        print("dirs",dirs)
        start_index = dirs.index(args.base_folder)
        dirname = os.path.join(*dirs[start_index+1:])
        outdir = os.path.join(args.out_folder, dirname)
        os.makedirs(outdir, exist_ok=True)
        
        # ----------------- load low-quality image ----------------- #
        pil_img = Image.open(file).convert("RGB")
        old_size = pil_img.size
        
        # ----------------- resize image to desired size ----------------- #
        if args.resize_mode == "square":
            pil_img = pil_img.resize((512, 512))
            warn(f"square mode, resize {old_size} -> {pil_img.size}")
        elif args.resize_mode == "auto":
            l = min(pil_img.size)
            if l < args.auto_mode_size:
                r = args.auto_mode_size / l
                pil_img = pil_img.resize(
                    tuple(math.ceil(x * r) for x in pil_img.size),
                    resample=Image.BICUBIC
                )
            warn(f"auto mode, resize {old_size} -> {pil_img.size}")
        elif args.resize_mode == "scale_auto":
            pil_img = pil_img.resize(
                tuple(math.ceil(x * args.auto_mode_scale) for x in pil_img.size),
                Image.BICUBIC
            )
            old_size = pil_img.size

            l = min(pil_img.size)
            if l < args.auto_mode_size:
                r = args.auto_mode_size / l
                pil_img = pil_img.resize(
                    tuple(math.ceil(x * r) for x in pil_img.size),
                    Image.BICUBIC
                )
            warn(f"scale_auto mode, scale = {args.auto_mode_scale}, resize {old_size} -> {pil_img.size}")
        else:
            warn(f"none mode, keep input size {old_size}")
        
        # ----------------- pad images ----------------- #
        h_before_pad, w_before_pad = pil_img.height, pil_img.width
        img = pad(np.array(pil_img), scale=64)
        h, w = img.shape[:2]
        
        for i in range(args.repeat_times):
            filename = os.path.basename(file)
            name, ext = filename.split(".")
            filename = f"{name}_{i}.{ext}"
            outpath = os.path.join(outdir, filename)
            
            if os.path.exists(outpath):
                assert args.skip_if_exist
                print(f"skip {outpath}")
                continue

            musiq = calculate_musiq(img)
            warn(f"low_quality_img_musiq: {musiq}")
            assert musiq >=0
            k = musiq/100.
            if k <= 0.4:
                k=0
                csft_w = (k-0.)/(0.7-0.)
            elif 0.4 < k <= 0.7:
                csft_w = (k-0.)/(0.7-0.)
            else:
                k=0.7
                csft_w = (k-0.)/(0.7-0.)
            
            print("k,csft_W:",k,csft_w)
            
            control_imgs = [((img / 255.0)*2.0 -1.0).astype(np.float32)] # [-1, 1], RGB, HWC

            seed_everything(-1)

            ddpm_preds = process(
                model, control_imgs, h, w, steps=args.steps,
                guess_mode=False, strength=1, scale=1,
                color_fix_type=args.color_fix_type,
                disable_preprocess_model=args.disable_preprocess_model,
                delta=k, csft_w=csft_w
            )
            control_img = ((control_imgs[0]/2.0 +0.5) * 255).clip(0, 255).astype(np.uint8)
            ddpm_pred = ddpm_preds[0]
            # if not args.disable_preprocess_model:
            #     denoiseAE_pred = get_denoiseAE_preds(model, control_imgs, delta=0)[0]
            # else:
            #     denoiseAE_pred = control_img
            
            control_img = control_img[:h_before_pad, :w_before_pad, :]
            ddpm_pred = ddpm_pred[:h_before_pad, :w_before_pad, :]
            # denoiseAE_pred = denoiseAE_pred[:h_before_pad, :w_before_pad, :]
            
            # resize back to input size
            def resize_back(img):
                return np.array(Image.fromarray(img).resize(
                    old_size, resample=Image.LANCZOS
                ))
            # if (w, h) != old_size and args.resize_back and args.resize_mode != "none":
            #     control_img, ddpm_pred, denoiseAE_pred = list(map(
            #         resize_back, [control_img, ddpm_pred, denoiseAE_pred]
            #     ))
            if (w, h) != old_size and args.resize_back and args.resize_mode != "none":
                control_img, ddpm_pred = list(map(
                    resize_back, [control_img, ddpm_pred]
                ))
            
            if args.show_lq:
                Image.fromarray(np.concatenate([control_img, denoiseAE_pred, ddpm_pred], axis=1)).save(outpath)
            else:
                Image.fromarray(ddpm_pred).save(outpath)
            warn(f"save to {outpath}")


if __name__ == "__main__":
    main()
