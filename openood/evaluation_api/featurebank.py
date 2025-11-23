import torch
from typing import Optional

class FeatureBank:
    value: Optional[torch.Tensor] = None

def feature_hook(module, inp, out):
    FeatureBank.value = inp[0].detach()
