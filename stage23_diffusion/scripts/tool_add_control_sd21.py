import sys
sys.path.append(".")
import os

assert len(sys.argv) == 4, 'Args are wrong.'

input_path = sys.argv[1]
output_path = sys.argv[2]
model_config = sys.argv[3]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from cldm.share import *
from cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]

model = create_model(config_path=model_config)
# here model_config is https://github.com/lllyasviel/ControlNet/blob/main/models/cldm_v21.yaml

pretrained_weights = torch.load(input_path)
# 从原来的SD Model里面把后缀相同的权重都复制到现在新建的model里面，即ControlModel，有SDModel的部分，和构建的ControlNet部分（line28）
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()
# 把模型初始化的权重读出来

target_dict = {}
for k in scratch_dict.keys():
    # 遍历模型里面的key，如果key带了前缀control_
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name #control_model vs model.diffusion_model
    else:
        copy_k = k

    if copy_k in pretrained_weights:
        control_shape = scratch_dict[k].shape
        pretrained_shape = pretrained_weights[copy_k].shape
        
        if control_shape == pretrained_shape:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            newly_added_channels = control_shape[1] - pretrained_shape[1]
            oc, _, h, w = pretrained_shape
            zero_weights = torch.zeros((oc, newly_added_channels, h, w)).type_as(pretrained_weights[copy_k])
            target_dict[k] = torch.cat((pretrained_weights[copy_k].clone(), zero_weights), dim=1)
            print(f"add zero weights to {copy_k} in pretrained weights, newly added channels = {newly_added_channels}")
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
