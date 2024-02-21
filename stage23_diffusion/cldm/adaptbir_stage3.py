import copy
from collections import OrderedDict
import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from cldm.spaced_sampler import SpacedSampler
import cldm.color_print as cp
from ldm.modules.spade import SPADE,STABLE_SPADE
from ldm.modules.diffusionmodules.util import normalization
import torch.nn.functional as F
import pyiqa
import time
import numpy as np

class Fuse_sft_block(nn.Module):

    def make_zero_conv(self, channels,dims=2):
        return zero_module(conv_nd(dims, channels, channels, 1, padding=0))
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.fd_norm = normalization(in_ch)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(in_ch, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.scale = nn.Conv2d(nhidden, out_ch, kernel_size=3, padding=1)

        self.shift = nn.Conv2d(nhidden, out_ch, kernel_size=3, padding=1)

    def forward(self, enc_feat, dec_feat, csft_w):
        enc_feat = self.mlp_shared(enc_feat)
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        bs = csft_w.shape[0]
        csft_w = csft_w.reshape(bs,1,1,1)
        residual = csft_w * (self.fd_norm(dec_feat) * scale + shift)
        out = dec_feat + residual
        return out

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


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ControlledSFTUnetModel(UNetModel):
    def __init__(
        self,
        dims=2,
        *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dims = dims
        # model_channels = self.model_channels
        input_chans = [self.model_channels]
        ch = self.model_channels
        for level, mult in enumerate(self.channel_mult):
            for nr in range(self.num_res_blocks[level]):
                ch = mult * self.model_channels
                input_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                input_chans.append(ch)
        # input_chans.append(ch)
       
        output_chans = [ch]
        self.Upsample_chans = []
        self.upsample_layers = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_chans.pop()
                output_chans.append(self.model_channels * mult)
                ch = self.model_channels * mult
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    self.Upsample_chans.append(out_ch)
                    self.upsample_layers.append(Upsample(ich, self.conv_resample, dims=dims, out_channels=out_ch))

        concat_chans = output_chans[::-1]
        self.cft_blocks = nn.ModuleList(
            [
                Fuse_sft_block(concat_chans[-1], concat_chans.pop())
            ]
        )
        for i in range(len(concat_chans)):
            self.cft_blocks.append(Fuse_sft_block(concat_chans[-1],concat_chans.pop()))

        for param in self.cft_blocks.parameters():
            param.requires_grad = True

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, control_feature=None, csft_w=None,**kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        idx= 0
        Upsample_times = 0
        if control is not None:
            h += control.pop()
            h = self.cft_blocks[idx](control_feature.pop(),h,csft_w)
            idx +=1

        for i, module in enumerate(self.output_blocks):
            # output_blocks are the decoder
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            # if not only middle control,then the control will also work on the decoder
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)
            if ((i+1)%3) == 0 and i != len(self.output_blocks)-1 :
                control_feature[-1]=self.upsample_layers[Upsample_times](control_feature[-1])
                Upsample_times +=1
            h = self.cft_blocks[idx](control_feature.pop(),h,csft_w)
            idx +=1

        h = h.type(x.dtype)
        return self.out(h)

# ControlNet just the orignal SDModel's encoder
# please check the https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py -->UNetModel
# It's almost the same.
class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels, #default=4
            model_channels, #default=320
            hint_channels, #default=4
            num_res_blocks, #default=2
            attention_resolutions, #default= [ 4, 2, 1 ]
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        x = torch.cat((x, hint), dim=1)
        outs = []
        control_features = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            control_features.append(h)
            outs.append(zero_conv(h, emb, context))
        h = self.middle_block(h, emb, context)
        control_features.append(h)
        outs.append(self.middle_block_out(h, emb, context))

        return outs,control_features
    
class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control,
                 preprocess_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ----------------- Control Module ----------------- #
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        
        # ----------------- Preprocess Module ----------------- #
        self.preprocess_model = instantiate_from_config(preprocess_config)
        # the preprocess_config is about the stage 1 model,e.g. SwinIR
        cp.warn(f"load preprocess model from {preprocess_config.ckpt}")
        ckpt = torch.load(preprocess_config.ckpt)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        reload_model(self.preprocess_model, ckpt)
        self.denoise_encoder: nn.Module = None
        self.musiq_model = pyiqa.create_metric('musiq',as_loss=False)
        self.musiq_model.eval()

    def init_denoise_encoder(self):
        cp.warn("make a copy of preprocess_model's encoder and frozen it")
        self.denoise_encoder = copy.deepcopy(self.preprocess_model.encoder)

        self.denoise_encoder.eval()
        self.denoise_encoder.train = disabled_train
        for p in self.denoise_encoder.parameters():
            p.requires_grad = False

    def apply_denoise_encoder(self, control):
        temp = self.denoise_encoder(control)
        moments = self.first_stage_model.quant_conv(temp)
        posterior = DiagonalGaussianDistribution(moments)

        return posterior

    # before here is the main point 
    @torch.no_grad()
    def get_input(self, batch, k, sample_posterior=False, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key] #hint
        if bs is not None:
            control = control[:bs]
        print("in get input, self.device is:",self.device)
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control

        self.musiq_model.to(self.device).eval()

        k_list = []
        csft_w_list = []
        for i in range(0,control.shape[0]):
            img = control[i]
            musiq_score = float(self.musiq_model(img[None,...]).detach().cpu().numpy())
            k = musiq_score/100.
            if k <= 0.4:
                k=0
                csft_w = (k-0.)/(0.7-0.)
            elif 0.4 < k <= 0.7:
                csft_w = (k-0.)/(0.7-0.)
            else:
                k=0.7
                csft_w = (k-0.)/(0.7-0.)
            k_list.append(k)
            csft_w_list.append(csft_w)


        batch_k = torch.tensor(k_list).reshape(control.shape[0],1,1,1).to(self.device)
        batch_csft_w = torch.tensor(csft_w_list).reshape(control.shape[0],1,1,1).to(self.device)

        posterior = self.apply_denoise_encoder(control)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        c_latent1 = self.first_stage_model.post_quant_conv(z)

        posterior2 = self.first_stage_model.encode(control).mode()
        c_latent2 = self.first_stage_model.post_quant_conv(posterior2)
        ones = torch.ones_like(batch_k)

        c_latent = c_latent1*((ones-batch_k)) +c_latent2*batch_k
        control_output = self.first_stage_model.decoder(c_latent)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control_output],middle=[z],csft_w=[batch_csft_w])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        csft_w = cond['csft_w']

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control, feature = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            feature = [f * scale for f, scale in zip(feature, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, control_feature=feature, only_mid_control=self.only_mid_control,csft_w=csft_w[0])

        return eps
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        # 
        csft_w = c["csft_w"][0]

        middle = c["middle"][0][:N]
        c_lq = c["lq"][0][:N]
        c_latent = c["c_latent"][0][:N]
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat
        log["lq"] = c_lq
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        # get denoise row
        samples, z_denoise_row = self.sample_log(
            cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent], "csft_w": [csft_w]},
            batch_size=N, steps=ddim_steps, eta=ddim_eta
        )
        x_samples = self.decode_first_stage(samples)
        log["samples"] = x_samples
        if plot_denoise_rows:
            denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
            log["denoise_row"] = denoise_grid

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, steps, **kwargs):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = sampler.sample(
            steps, batch_size, shape, cond, unconditional_guidance_scale=1.0,
            unconditional_conditioning=None
        )
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        # during stage3 training, we only optimize the CSFT layers.
        for name,param in self.model.named_parameters():
            if 'cft' in name:
                params.append(param)
                # print(name)
            if 'upsample_layers' in name:
                params.append(param)
                # print(name)
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
