import os
from omegaconf import OmegaConf
import torch
import importlib
from argparse import ArgumentParser
from ldm.util import instantiate_from_config
import torchvision.models as models
from thop import profile
from typing import Mapping, Any

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    args = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    model = instantiate_from_config(config.model)

    input = torch.randn(1,3,256,256)
    # output = model(input)
    # print(output.shape)

    flops, params = profile(model,inputs=(input, ))
    print('Flops:',flops)
    print('Parameters:',params)


if __name__ == "__main__":
    main()
