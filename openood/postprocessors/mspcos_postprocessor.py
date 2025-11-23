# openood/postprocessors/mspcos_postprocessor.py
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
from openood.evaluation_api.featurebank import feature_hook, FeatureBank


class MSPCosPostprocessor(BasePostprocessor):
    """
    ID-confidence (ID-high):
      score(x) = MSP(x) + mean_i |cos(z, w_i)|
      where z is the penultimate feature (L2-normalized),
            w_i are classifier weights (L2-normalized rows).
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        pp = getattr(config, "postprocessor", None)

        def _get(key, default):
            try:
                if pp is None: return default
                if hasattr(pp, key): return getattr(pp, key)
                if hasattr(pp, "args") and hasattr(pp.args, key): return getattr(pp.args, key)
                if isinstance(pp, dict):
                    if key in pp: return pp[key]
                    if "args" in pp and key in pp["args"]: return pp["args"][key]
            except Exception:
                pass
            return default

        self.eps = float(_get("eps", 1e-12))

        self._hook_handle = None
        self.cls_weight_norm = None
        self._feat_dim = None

    @staticmethod
    def _find_feat_layer(net: nn.Module) -> nn.Module:
        # 选择紧邻分类器的层，避免 4D->随机投影
        for name in ("avgpool", "global_pool", "pool", "head"):
            if hasattr(net, name):
                return getattr(net, name)
        children = list(net.children())
        return children[-2] if len(children) >= 2 else net

    @staticmethod
    def _find_classifier_weight(net: nn.Module) -> torch.Tensor:
        # 定位最终线性分类器权重
        for name in ("fc", "linear", "classifier", "head", "last_linear"):
            if hasattr(net, name):
                mod = getattr(net, name)
                if isinstance(mod, nn.Linear) and hasattr(mod, "weight"):
                    return mod.weight
        for _, mod in net.named_modules():
            if isinstance(mod, nn.Linear) and hasattr(mod, "weight"):
                return mod.weight
        raise AttributeError("MSPCos: cannot locate final nn.Linear.weight")

    def setup(self, net: nn.Module, id_loader: Any, ood_loader_dict: Any = None) -> None:
        # 注册 hook（实例内管理）
        feat_layer = self._find_feat_layer(net)
        if self._hook_handle is None:
            self._hook_handle = feat_layer.register_forward_hook(feature_hook)

        # 缓存并归一化分类器权重
        device = next(net.parameters()).device
        w = self._find_classifier_weight(net).detach()           # [C, D]
        self.cls_weight_norm = F.normalize(w, dim=1, eps=self.eps).to(device)
        self._feat_dim = int(w.shape[1])

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = net(data)  # 触发 hook

        if FeatureBank.value is None:
            raise RuntimeError("[mspcos] FeatureBank empty; ensure setup() and a forward pass occurred.")

        # 特征拉直并 L2 归一化
        feat = FeatureBank.value.detach().view(data.size(0), -1)  # [B, D?]
        feat = F.normalize(feat, p=2, dim=1, eps=self.eps)

        # 维度一致性检查（如果仍不一致，做一次安全降维/截断，而不是每次随机投影）
        if feat.shape[1] != self._feat_dim:
            # 尽量在选对层后不会发生；这里做防御性对齐：截断或零填充
            D = self._feat_dim
            if feat.shape[1] > D:
                feat = feat[:, :D]
            else:
                pad = D - feat.shape[1]
                feat = torch.cat([feat, feat.new_zeros(feat.size(0), pad)], dim=1)

        # MSP
        probs = F.softmax(logits, dim=1)
        msp = probs.max(dim=1).values

        # 平均 |cos|
        cos_all = torch.matmul(feat, self.cls_weight_norm.t()).abs()  # [B, C]
        mean_cos = cos_all.mean(dim=1)

        score = msp + mean_cos            # ID-high
        pred = probs.argmax(dim=1)
        return pred, score

    def teardown(self):
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None

    def __del__(self):
        self.teardown()
