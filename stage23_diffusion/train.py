from argparse import ArgumentParser

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cldm.share import *
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, instantiate_from_config
from cldm.color_print import warn
from cldm.cldm import reload_model
from os import path as osp
import os


def main(opt):
    
    print(f"{'- ' * 20}\nconfig: \n{opt}\n{'- ' * 20}\n")
    pl.seed_everything(opt.train.seed)
    
    model = create_model(opt.model.config).cpu()
    if opt.model.auto_resume:
        state_path = osp.join(opt.expdir, 'lightning_logs')
        version_names = os.listdir(state_path)
        versions = [int(version_name.split("_")[1]) for version_name in version_names]
        latest_version_name = f"version_{max(versions)}"
        state_path = osp.join(state_path, latest_version_name,'checkpoints')
        ckpt_names = os.listdir(state_path)
        assert len(ckpt_names) == 1
        opt.model.resume = osp.join(state_path, ckpt_names[0])
    model.load_state_dict(load_state_dict(opt.model.resume, location='cpu'), strict=False)
    model.learning_rate = opt.train.learning_rate
    model.sd_locked = opt.train.sd_locked
    model.only_mid_control = opt.train.only_mid_control

    if hasattr(model, "denoise_encoder"):
        warn(f"initialize denoise encoder for this model")
        model.init_denoise_encoder()
    
    dataset = instantiate_from_config(opt.data)
    dataloader = DataLoader(dataset, num_workers=opt.data.num_workers, prefetch_factor=opt.train.prefetch_factor,
                            batch_size=opt.train.batch_size, shuffle=True)
    
    logger = ImageLogger(batch_frequency=opt.train.logger_freq)
    checkpoint_callback = ModelCheckpoint(
        filename="{step}",
        every_n_train_steps=opt.train.save_every_n_steps
    )
    
    trainer = pl.Trainer(accelerator="ddp", precision=32, callbacks=[logger, checkpoint_callback],
                         gpus=opt.train.gpus, default_root_dir=opt.expdir, max_steps=opt.train.max_steps)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    
    main(OmegaConf.load(args.cfg))
