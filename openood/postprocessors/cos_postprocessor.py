# openood/postprocessors/cos_postprocessor.py
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
from openood.evaluation_api.featurebank import feature_hook, FeatureBank


class CosinePostprocessor(BasePostprocessor):
    """
    Prototype cosine similarity (ID-high):
      score = max_c cos(f, μ_c), larger => more ID-like
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
        self.class_means = None
        self._valid_mask = None
        self._feat_dim = None

    @staticmethod
    def _find_feat_layer(net: nn.Module) -> nn.Module:
        for name in ("avgpool", "global_pool", "pool", "head"):
            if hasattr(net, name):
                return getattr(net, name)
        children = list(net.children())
        return children[-2] if len(children) >= 2 else net

    @torch.no_grad()
    def _build_class_means(
        self, net: nn.Module, feat_layer: nn.Module, id_loader: torch.utils.data.DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._hook_handle is None:
            self._hook_handle = feat_layer.register_forward_hook(feature_hook)

        device = next(net.parameters()).device
        was_training = net.training
        net.eval()

        num_classes = getattr(getattr(id_loader, "dataset", None), "num_classes", None)
        sum_feat, count = None, None

        for batch in id_loader:
            if isinstance(batch, dict):
                data, labels = batch["data"], batch["label"]
            else:
                data, labels = batch[:2]
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            _ = net(data)
            if FeatureBank.value is None:
                raise RuntimeError("[cos] FeatureBank empty; hook not triggered.")
            feat = FeatureBank.value.detach().view(data.size(0), -1)
            feat = F.normalize(feat, p=2, dim=1, eps=self.eps)

            if sum_feat is None:
                self._feat_dim = int(feat.size(1))
                C = int(num_classes) if num_classes is not None else int(labels.max().item()) + 1
                sum_feat = torch.zeros(C, self._feat_dim, device=device, dtype=feat.dtype)
                count = torch.zeros(C, device=device, dtype=torch.long)

            if num_classes is None:
                need_c = int(labels.max().item()) + 1
                if need_c > sum_feat.size(0):
                    pad = need_c - sum_feat.size(0)
                    sum_feat = torch.cat([sum_feat, torch.zeros(pad, self._feat_dim, device=device, dtype=feat.dtype)], 0)
                    count = torch.cat([count, torch.zeros(pad, device=device, dtype=torch.long)], 0)

            for c in labels.unique():
                ci = int(c.item())
                m = (labels == ci)
                if m.any():
                    sum_feat[ci] += feat[m].sum(0)
                    count[ci] += int(m.sum().item())

        valid = count > 0
        class_means = torch.zeros_like(sum_feat)
        if valid.any():
            class_means[valid] = sum_feat[valid] / count[valid].unsqueeze(1)
        class_means = F.normalize(class_means, p=2, dim=1, eps=self.eps)

        if was_training:
            net.train()
        return class_means, valid

    def setup(self, net: nn.Module, id_loader: Any, ood_loader_dict: Any = None) -> None:
        if isinstance(id_loader, dict):
            id_loader = id_loader.get("train", next(iter(id_loader.values())))
        feat_layer = self._find_feat_layer(net)
        self.class_means, self._valid_mask = self._build_class_means(net, feat_layer, id_loader)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = net(data)
        if FeatureBank.value is None:
            raise RuntimeError("[cos] FeatureBank empty during postprocess.")
        feat = FeatureBank.value.detach().view(data.size(0), -1)
        feat = F.normalize(feat, p=2, dim=1, eps=self.eps)

        cm = self.class_means if self._valid_mask is None else self.class_means[self._valid_mask]
        sim = feat @ cm.t()                  # [B, C]
        max_sim, _ = sim.max(dim=1)
        score = max_sim                      # ID-high
        pred = torch.argmax(logits, dim=1)
        return pred, score

    def teardown(self):
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            finally:
                self._hook_handle = None

    def __del__(self):
        self.teardown()
