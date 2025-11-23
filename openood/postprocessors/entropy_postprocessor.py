# from typing import Any

# import numpy as np
# import torch
# import torch.nn as nn
# from tqdm import tqdm

# from .base_postprocessor import BasePostprocessor

# class EntropyPostprocessor(BasePostprocessor):
#     @torch.no_grad()
#     def postprocess(self, net: nn.Module, data: Any):
#         logits = net(data)
#         prob   = torch.softmax(logits, dim=1)
#         ent    = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
#         pred   = torch.argmax(prob, dim=1)
#         return pred, -ent           



from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor

class EntropyPostprocessor(BasePostprocessor):
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = net(data)                              # [B, C]
        log_prob = F.log_softmax(logits, dim=1)        # 稳定
        prob = log_prob.exp()
        ent = -(prob * log_prob).sum(dim=1)            # 香农熵
        score = -ent                                    # ID-high / OOD-low
        pred = torch.argmax(logits, dim=1)
        return pred, score
