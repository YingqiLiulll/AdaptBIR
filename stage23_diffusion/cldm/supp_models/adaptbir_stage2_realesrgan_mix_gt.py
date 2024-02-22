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
import numpy


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
    model.load_state_dict(ckpt)


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            # output_blocks are the decoder
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            # if not only middle control,then the control will also work on the decoder
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

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
            # self.num_res_blocks = [2, 2, 2, 2]
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
        # notice！
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        # we need an encoder to map input to 64x64 dimension, we use concat
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        # we delete the self.input_hint_block
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            # level:[0,1,2,3]
            # mult:[1,2,4,4]
            for nr in range(self.num_res_blocks[level]):
                # nr:[0,1]
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
                # every time input_block &zero_convs will append together
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                # if it's not the last level
                # add another ResBlock which has the same input_channel with out_channels
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
                # ds:[1,2,4,8,16,32,64,128,256]
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
        # timestep process keep the same
        # print("x.shape",x.shape)
        # print("hint.shape",hint.shape)
        x = torch.cat((x, hint), dim=1)
        # we use concat to sew the x and hint，in ControlNet they directly add.
        # we omit self.input_hint_block
        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))
            # this is for the last level
        # directly to the middle block
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        # so in fact, the out of middle_block is the first layer output

        return outs

# use ControlLDM to copy the parameters
class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, 
                 preprocess_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ----------------- Control Module ----------------- #
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        # 继承自LDM，这个control_model可以复制原来的LDM
        self.control_key = control_key
        # control key:[default] "hint"
        self.only_mid_control = only_mid_control
        # only_mid_control：【default] false
        self.control_scales = [1.0] * 13
        # control_scales: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        # ----------------- Preprocess Module ----------------- #
        self.preprocess_model = instantiate_from_config(preprocess_config)
        # the preprocess_config is about the stage 1 model,e.g. SwinIR
        cp.warn(f"load preprocess model from {preprocess_config.ckpt}")
        ckpt = torch.load(preprocess_config.ckpt)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        reload_model(self.preprocess_model, ckpt)
        self.denoise_encoder: nn.Module = None
        # self.cond_encoder: nn.Module = None

    def init_denoise_encoder(self):
        cp.warn("make a copy of preprocess_model's encoder and frozen it")
        self.denoise_encoder = nn.Sequential(
            copy.deepcopy(self.preprocess_model.encoder),
            copy.deepcopy(self.preprocess_model.quant_conv))
        # print("self.denoise_encoder:",self.denoise_encoder)
        self.denoise_post_quant_conv = copy.deepcopy(self.preprocess_model.post_quant_conv)
        self.denoise_decoder = copy.deepcopy(self.preprocess_model.decoder)

        self.denoise_encoder.eval()
        self.denoise_post_quant_conv.eval()
        self.denoise_decoder.eval()

        self.denoise_encoder.train = disabled_train
        self.denoise_post_quant_conv.train = disabled_train
        self.denoise_decoder.train = disabled_train

        for p in self.denoise_encoder.parameters():
            p.requires_grad = False
        for p in self.denoise_post_quant_conv.parameters():
            p.requires_grad = False
        for p in self.denoise_decoder.parameters():
            p.requires_grad = False

    def apply_denoise_encoder(self, control):
        moments = self.denoise_encoder(control)
        posterior = DiagonalGaussianDistribution(moments)

        return posterior
    
    def apply_gt_encoder(self, control):
        gt_h = self.first_stage_model.encoder(control)
        gt_z = self.first_stage_model.quant_conv(gt_h)
        control_posterior = DiagonalGaussianDistribution(gt_z)

        return control_posterior

    # before here is the main point 
    @torch.no_grad()
    def get_input(self, batch, k, sample_posterior=False, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        # x is z,and c is condition
        control = batch[self.control_key] #hint
        type = batch["type"]
        if bs is not None:
            control = control[:bs]
        if bs is not None:
            type = type[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        x_realesrgan_list = []
        x_gt_list = []
        realesrgan_control_list = []
        gt_control_list = []
        # control_posterior = self.apply_denoise_encoder(control)
        for i in range(len(type)):
            if type[i] == 'realesrgan':
                realesrgan_control_list.append(control[i])
                x_realesrgan_list.append(x[i,:])
            elif type[i] == 'gtpaired':
                gt_control_list.append(control[i])
                x_gt_list.append(x[i,:])

        if realesrgan_control_list and gt_control_list:
            realesrgan_control = torch.stack(realesrgan_control_list, axis=0)
            gt_control = torch.stack(gt_control_list, axis=0)
            lq = torch.cat((realesrgan_control,gt_control),dim=0)
            realesrgan_control_z = self.apply_denoise_encoder(realesrgan_control).mode()
            gt_control_z = self.apply_gt_encoder(gt_control).mode()
            z= torch.cat((realesrgan_control_z,gt_control_z),dim=0)

        elif realesrgan_control_list and (not gt_control_list):
            print("gt_control_list is none!")
            realesrgan_control = torch.stack(realesrgan_control_list, axis=0)
            lq = realesrgan_control
            realesrgan_control_z = self.apply_denoise_encoder(realesrgan_control).mode()
            z = realesrgan_control_z

        elif gt_control_list and (not realesrgan_control_list):
            print("realesrgan_control_list is none!")
            gt_control = torch.stack(gt_control_list, axis=0)
            lq = gt_control
            gt_control_z = self.apply_gt_encoder(gt_control).mode()
            z = gt_control_z
        
        x_list = x_realesrgan_list + x_gt_list
        x = torch.stack(x_list, axis=0)
        c_latent = self.denoise_post_quant_conv(z)
        control_output = self.denoise_decoder(c_latent)
        # print("c_latent.shape",c_latent.shape)
        # here the control is the image after SwinIR's preprocess
        # apply condition encoder
        # c_latent = self.apply_condition_encoder(control)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], lq=[lq], c_concat=[control_output],middle=[z])
    # In fact, in this repo, all we need is the c_latent.the c_concat is useless

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps
        # eps是diffusion model预测出的噪声

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
        middle = c["middle"][0][:N]
        c_lq = c["lq"][0][:N]
        c_latent = c["c_latent"][0][:N]
        # we add here
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat
        # log["control_latent"] = self.decode_first_stage(middle)
        log["lq"] = c_lq
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        # get denoise row
        samples, z_denoise_row = self.sample_log(
            cond={"c_concat": [c_cat], "c_crossattn": [c], "c_latent": [c_latent]},
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
        # TODO: make condition encoder trainable ?
        params = list(self.control_model.parameters())
        # during training, we only optimize the control_model
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
