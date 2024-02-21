import torch
import torch.nn as nn

class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, reconstructions, split="train"):
        l2_loss = torch.square(inputs.contiguous() - reconstructions.contiguous())
        loss = l2_loss
        log = {"{}/l2_loss".format(split): loss.detach().mean()}
        return loss, log

