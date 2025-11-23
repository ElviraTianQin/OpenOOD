# openood/postprocessors/norm_postprocessor.py
from typing import Any, Optional
import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from openood.evaluation_api.featurebank import feature_hook, FeatureBank


class NormPostprocessor(BasePostprocessor):
    """
    ID-confidence by feature L2 norm (ID-high, OOD-low).

    - setup(): registers a forward hook on the penultimate layer to capture features.
    - postprocess(): score = ||feat||_2 (larger => more likely ID).
    - teardown(): removes the hook to avoid residual side-effects.
    """

    def __init__(self, config, **kwargs):
        """
        OpenOOD 工厂会以 `NormPostprocessor(config)` 的形式实例化，
        因此这里必须接收 config，并传给 BasePostprocessor。
        """
        super().__init__(config)

        # 这里你可以按需要从 config 中取参数（当前版本没额外参数）
        # 例子（可留空）：
        # pp = getattr(config, "postprocessor", None)
        # if pp is not None and hasattr(pp, "args"):
        #     self.some_param = getattr(pp.args, "some_param", default_value)

        # 初始化实例状态
        self._hook_handle = None


    @staticmethod
    def _find_feat_layer(net: nn.Module) -> nn.Module:
        # Prefer common attribute names; fallback to penultimate child.
        for name in ("avgpool", "global_pool", "pool", "head"):
            if hasattr(net, name):
                return getattr(net, name)
        children = list(net.children())
        if len(children) < 2:
            return net  # extreme fallback
        return children[-2]

    def setup(
        self,
        net: nn.Module,
        id_loader_dict: Any = None,
        ood_loader_dict: Any = None,
    ) -> None:
        feat_layer = self._find_feat_layer(net)
        if self._hook_handle is None:
            self._hook_handle = feat_layer.register_forward_hook(feature_hook)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = net(data)  # forward pass triggers feature_hook -> FeatureBank.value

        if FeatureBank.value is None:
            raise RuntimeError(
                "[norm] FeatureBank is empty. Ensure setup() was called and a forward pass occurred."
            )

        feat = FeatureBank.value.detach().view(data.size(0), -1)  # [B, D]
        # ID-high score: larger L2 norm => more ID-like
        score = torch.linalg.vector_norm(feat, ord=2, dim=1)
        pred = torch.argmax(logits, dim=1)
        return pred, score

    def teardown(self):
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None

    def __del__(self):
        # Best-effort cleanup in case caller didn't call teardown()
        self.teardown()
