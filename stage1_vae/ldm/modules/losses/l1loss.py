import torch
import torch.nn as nn
from torch.nn import functional as F
from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class L1WithLPIPS(nn.Module):
    def __init__(self,perceptual_weight=0.25, l1_weight=1.0, reduction:str = 'mean', disc_conditional=False):
        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.l1_weight = l1_weight
        self.reduction = reduction

    def forward(self, inputs, reconstructions, split="train"):
        lpips_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()).mean()
        l1_loss = F.l1_loss(inputs, reconstructions, reduction=self.reduction)
        loss = lpips_loss * self.perceptual_weight + l1_loss * self.l1_weight
        log = {"{}/training_loss".format(split): loss.detach().mean()}
        return loss, log

