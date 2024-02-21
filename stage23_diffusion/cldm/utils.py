import numpy as np
import torch
import cv2


def to_pil_image_array(inputs, mem_order, val_range, channel_order):
    # convert inputs to numpy array
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()
    assert isinstance(inputs, np.ndarray)
    
    # make sure that inputs is a 4-dimension array
    if mem_order in ["hwc", "chw"]:
        inputs = inputs[None, ...]
        mem_order = f"n{mem_order}"
    # to NHWC
    if mem_order == "nchw":
        inputs = inputs.transpose(0, 2, 3, 1)
    # to RGB
    if channel_order == "bgr":
        inputs = inputs[..., ::-1].copy()
    else:
        assert channel_order == "rgb"
    
    if val_range == "0,1":
        inputs = inputs * 255
    elif val_range == "-1,1":
        inputs = (inputs + 1) * 127.5
    else:
        assert val_range == "0,255"
    
    inputs = inputs.clip(0, 255).astype(np.uint8)
    return [inputs[i] for i in range(len(inputs))]

def put_text(pil_img_arr, text):
    cv_img = pil_img_arr[..., ::-1].copy()
    cv2.putText(cv_img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return cv_img[..., ::-1].copy()
