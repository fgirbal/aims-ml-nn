import torch
import numpy as np
from enum import Enum
from typing import Tuple


class InitializationType(Enum):
    ZEROS = 1
    XAVIER_NORMAL = 2


def calculate_fan_in_and_fan_out(tensor: torch.tensor) -> Tuple[int, int]:
    """Calculate fan in and fan out of a certain tensor. Adapted from:
    https://pytorch.org/docs/stable/_modules/torch/nn/init.html
    
    Args:
        tensor (torch.tensor): input tensor
    
    Returns:
        Tuple[int, int]: fan in an fan out values
    """
    dimensions = tensor.dim()

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)

    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_normal(tensor: torch.tensor, gain=1.0):
    """Initialize a tensor (in place) given a Xavier normal. Adapted from:
    https://pytorch.org/docs/stable/_modules/torch/nn/init.html
    
    Args:
        tensor (torch.tensor): input tensor
        gain (float, optional): parameter of the Xavier normal
    """
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))

    with torch.no_grad():
        tensor.normal_(0, std)

