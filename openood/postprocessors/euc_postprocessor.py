# from typing import Any
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .base_postprocessor import BasePostprocessor
# from openood.evaluation_api.featurebank import feature_hook, FeatureBank



# class EuclideanPostprocessor(BasePostprocessor):
#     def __init__(self, normalize=True, eps=1e-12):
#         self.normalize = normalize
#         self.eps = eps
#         self._hook_handle = None
#         self.class_means = None
#         self._feat_dim = None

#     @staticmethod
#     def _find_feat_layer(net: nn.Module) -> nn.Module:
#         # 更保守：优先找常见名，其次倒数第二层兜底
#         for name in ["avgpool", "global_pool", "pool", "head"]:  # 视模型而定
#             if hasattr(net, name):
#                 return getattr(net, name)
#         # 兜底
#         return list(net.children())[-2]

#     @torch.no_grad()
#     def _build_class_means(self, net, feat_layer, id_loader):
#         # 注册 hook
#         if self._hook_handle is None:
#             self._hook_handle = feat_layer.register_forward_hook(feature_hook)

#         device = next(net.parameters()).device
#         num_classes = int(getattr(id_loader.dataset, "num_classes"))
#         was_training = net.training
#         net.eval()  # 统计时用 eval，稳定 BN/Dropout

#         sum_feat = None
#         count = torch.zeros(num_classes, device=device, dtype=torch.long)

#         for batch in id_loader:
#             if isinstance(batch, dict):
#                 data, labels = batch["data"], batch["label"]
#             else:
#                 data, labels = batch[:2]
#             data, labels = data.to(device), labels.to(device)
#             _ = net(data)
#             feat = FeatureBank.value.detach()
#             feat = feat.view(feat.size(0), -1)  # [B, D]

#             if self.normalize:
#                 feat = F.normalize(feat, p=2, dim=1, eps=self.eps)

#             if sum_feat is None:
#                 self._feat_dim = feat.size(1)
#                 sum_feat = torch.zeros(num_classes, self._feat_dim, device=device)

#             # 累加到对应类
#             for c in labels.unique():
#                 m = (labels == c)
#                 sum_feat[c] += feat[m].sum(0)
#                 count[c] += int(m.sum())

#         # 构建均值，跳过计数为0的类
#         valid = count > 0
#         class_means = torch.zeros(num_classes, self._feat_dim, device=device)
#         class_means[valid] = sum_feat[valid] / count[valid].unsqueeze(1)

#         if self.normalize:
#             class_means = F.normalize(class_means, p=2, dim=1, eps=self.eps)

#         if was_training:
#             net.train()

#         return class_means, valid

#     def setup(self, net, id_loader, ood_loader_dict=None):
#         # if getattr(net, "_euclid_hook_registered", False):
#         #     return
#         if isinstance(id_loader, dict):
#             id_loader = id_loader.get("train", next(iter(id_loader.values())))

#         feat_layer = self._find_feat_layer(net)
#         self.class_means, self._valid_mask = self._build_class_means(net, feat_layer, id_loader)
#         # net._euclid_hook_registered = True

#     @torch.no_grad()
#     def postprocess(self, net, data):
#         logits = net(data)
#         feat = FeatureBank.value.detach().view(data.size(0), -1)

#         if self.normalize:
#             feat = F.normalize(feat, p=2, dim=1, eps=self.eps)

#         # 距离：若已归一化，可用余弦快速算
#         # dist^2 = 2 - 2 * cos(f, mu)
#         # 分数(越大越 OOD)：min_dist
#         sim = torch.matmul(feat, self.class_means.t())  # [B, C]
#         sim = sim[:, self._valid_mask] if hasattr(self, "_valid_mask") else sim
#         # 数值安全：裁剪到[-1,1]
#         sim = sim.clamp(-1.0, 1.0)
#         dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))  # [B, C]
#         min_dist, _ = dist.min(dim=1)

#         score = min_dist  # 越大越 OOD
#         pred = torch.argmax(logits, dim=1)
#         return pred, score

#     def teardown(self):
#         if self._hook_handle is not None:
#             self._hook_handle.remove()
#             self._hook_handle = None
#


####### 上述是 10.20最后一版  得分方向是反的



# openood/postprocessors/euc_postprocessor.py
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
from openood.evaluation_api.featurebank import feature_hook, FeatureBank


class EuclideanPostprocessor(BasePostprocessor):
    """
    Prototype-distance OOD: score = -min Euclidean distance to class means (ID-high).

    - setup(): build per-class prototypes from ID train loader.
    - postprocess(): score = -min_dist (larger => more ID-like).
    - teardown(): remove hook to avoid residual side-effects.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        pp = getattr(config, "postprocessor", None)

        def _get(key, default):
            # 支持 postprocessor.xx 或 postprocessor.args.xx 两种写法
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

        self.normalize = bool(_get("normalize", True))
        self.eps = float(_get("eps", 1e-12))

        self._hook_handle = None
        self.class_means = None
        self._valid_mask = None
        self._feat_dim = None

    # ---------------- utils ----------------
    @staticmethod
    def _find_feat_layer(net: nn.Module) -> nn.Module:
        for name in ("avgpool", "global_pool", "pool", "head"):
            if hasattr(net, name):
                return getattr(net, name)
        children = list(net.children())
        return children[-2] if len(children) >= 2 else net

    def _ensure_capacity(self,
                         sum_feat: torch.Tensor,
                         count: torch.Tensor,
                         need_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cur_c = sum_feat.size(0)
        if need_classes <= cur_c:
            return sum_feat, count
        device, dtype = sum_feat.device, sum_feat.dtype
        feat_dim = sum_feat.size(1)
        pad = need_classes - cur_c
        sum_feat = torch.cat([sum_feat, torch.zeros(pad, feat_dim, device=device, dtype=dtype)], dim=0)
        count = torch.cat([count, torch.zeros(pad, device=device, dtype=count.dtype)], dim=0)
        return sum_feat, count

    @torch.no_grad()
    def _build_class_means(self,
                           net: nn.Module,
                           feat_layer: nn.Module,
                           id_loader: torch.utils.data.DataLoader
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hook（仅本实例持有）
        if self._hook_handle is None:
            self._hook_handle = feat_layer.register_forward_hook(feature_hook)

        device = next(net.parameters()).device
        was_training = net.training
        net.eval()

        ds_num_classes = getattr(getattr(id_loader, "dataset", None), "num_classes", None)

        sum_feat: Optional[torch.Tensor] = None
        count: Optional[torch.Tensor] = None

        for batch in id_loader:
            # 支持 dict / tuple
            if isinstance(batch, dict):
                data, labels = batch["data"], batch["label"]
            else:
                data, labels = batch[:2]

            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            _ = net(data)  # feature_hook -> FeatureBank.value
            if FeatureBank.value is None:
                raise RuntimeError("[euc] FeatureBank is empty; hook not triggered.")
            feat = FeatureBank.value.detach().view(data.size(0), -1)

            if self.normalize:
                feat = F.normalize(feat, p=2, dim=1, eps=self.eps)

            if sum_feat is None:
                self._feat_dim = int(feat.size(1))
                C = int(ds_num_classes) if ds_num_classes is not None else (int(labels.max().item()) + 1)
                sum_feat = torch.zeros(C, self._feat_dim, device=device, dtype=feat.dtype)
                count = torch.zeros(C, device=device, dtype=torch.long)

            if ds_num_classes is None:
                need_c = int(labels.max().item()) + 1
                sum_feat, count = self._ensure_capacity(sum_feat, count, need_c)

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
        if self.normalize:
            class_means = F.normalize(class_means, p=2, dim=1, eps=self.eps)

        if was_training:
            net.train()

        return class_means, valid

    # ---------------- lifecycle ----------------
    def setup(self,
              net: nn.Module,
              id_loader: Any,
              ood_loader_dict: Any = None) -> None:
        if isinstance(id_loader, dict):
            id_loader = id_loader.get("train", next(iter(id_loader.values())))
        feat_layer = self._find_feat_layer(net)
        self.class_means, self._valid_mask = self._build_class_means(net, feat_layer, id_loader)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if self.class_means is None:
            raise RuntimeError("[euc] class_means is None; call setup() first.")

        logits = net(data)  # hook 写入 FeatureBank.value
        if FeatureBank.value is None:
            raise RuntimeError("[euc] FeatureBank is empty during postprocess.")

        feat = FeatureBank.value.detach().view(data.size(0), -1)

        if self.normalize:
            feat = F.normalize(feat, p=2, dim=1, eps=self.eps)
            # 归一化后用 cos->euclidean，快且稳
            cm = self.class_means
            if self._valid_mask is not None:
                cm = cm[self._valid_mask]
            sim = feat @ cm.t()                   # [B, C]
            sim = sim.clamp(-1.0, 1.0)
            dist = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))
        else:
            cm = self.class_means
            if self._valid_mask is not None:
                cm = cm[self._valid_mask]
            dist = torch.cdist(feat, cm)         # [B, C], 真实欧氏

        min_dist, _ = dist.min(dim=1)
        # 统一到“ID 高、OOD 低”方向：
        score = -min_dist
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
        self.teardown()
