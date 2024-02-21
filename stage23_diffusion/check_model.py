import torch
from torchsummary import summary
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import pytorch_lightning as pl
from memory import ModelSummary


configs = OmegaConf.load('diffusion/configs/model_stage2.yaml')
model = instantiate_from_config(configs.model)
model.configs = configs
print("model",model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
results = ModelSummary(model, mode='top')
print(results)
