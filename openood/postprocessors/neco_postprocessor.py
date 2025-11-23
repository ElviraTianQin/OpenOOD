# from typing import Any
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.decomposition import PCA

# from .base_postprocessor import BasePostprocessor
# from openood.evaluation_api.featurebank import feature_hook, FeatureBank


# class NecoPostprocessor(BasePostprocessor):
#     @staticmethod
#     def _find_feat_layer(net: nn.Module) -> nn.Module:
#         if hasattr(net, "avgpool"):
#             return net.avgpool
#         return list(net.children())[-2]

#     @torch.no_grad()
#     def setup(self,
#               net: nn.Module,
#               id_loader: Any,
#               ood_loader_dict=None) -> None:
#         if getattr(net, "_neco_hook_registered", False):
#             return  # already set up

#         feat_layer = self._find_feat_layer(net)

#         # If id_loader is a dict (e.g., {"train": train_loader, ...}), pick train_loader
#         if isinstance(id_loader, dict):
#             id_loader = id_loader.get("train", next(iter(id_loader.values())))

#         # Register the hook so that FeatureBank.value is filled after each forward pass
#         feat_layer.register_forward_hook(feature_hook)

#         device = next(net.parameters()).device
#         all_feats = []  # will collect torch.Tensor feature batches on CPU

#         # Step 1: loop over all ID data and collect features
#         for batch in id_loader:
#             # Support both dict-based and tuple/list-based batches
#             if isinstance(batch, dict):
#                 data = batch["data"]
#             else:
#                 data = batch[0]

#             data = data.to(device)
#             _ = net(data)  # forward → feature_hook stores features into FeatureBank.value
#             batch_feat = FeatureBank.value.detach().cpu().view(data.size(0), -1)
#             all_feats.append(batch_feat)

#         # Concatenate all features along dimension 0 → shape [N_id, D]
#         all_feats_tensor = torch.cat(all_feats, dim=0)
#         all_feats_np = all_feats_tensor.numpy()  # convert to numpy for PCA

#         # Step 2: fit PCA with n_components = full feature dimension
#         self._pca = PCA(n_components=all_feats_np.shape[1])
#         self._pca.fit(all_feats_np)

#         # Mark that we've registered the hook and fitted PCA
#         net._neco_hook_registered = True

#     @torch.no_grad()
#     def postprocess(self, net: nn.Module, data: Any):
#         # 1) Forward pass, predicts logits and updates FeatureBank.value
#         logits = net(data)

#         # 2) Extract, flatten, and normalize features
#         feat: torch.Tensor = FeatureBank.value.detach().view(data.size(0), -1)
#         feat_normed = F.normalize(feat, p=2, dim=1)

#         # Convert normalized features to numpy for PCA projection
#         h_np = feat_normed.cpu().numpy()
#         h_pca = self._pca.transform(h_np)  # shape [B, D]

#         # Compute L2 norms
#         norms_pca = (h_pca ** 2).sum(axis=1) ** 0.5
#         norms_orig = (h_np ** 2).sum(axis=1) ** 0.5

#         # Avoid division by zero: add a tiny epsilon to denominators
#         eps = 1e-12
#         ratio = norms_pca / (norms_orig + eps)

#         # NECO score = - ratio (so that larger score indicates more OOD-like)
#         score = torch.from_numpy(ratio).to(feat.device)

#         # Prediction remains standard argmax on logits
#         pred = torch.argmax(logits, dim=1)
#         return pred, score




# openood/postprocessors/neco_postprocessor.py
# from typing import Any, Optional, Tuple
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.decomposition import PCA

# from .base_postprocessor import BasePostprocessor
# from openood.evaluation_api.featurebank import feature_hook, FeatureBank


# class NecoPostprocessor(BasePostprocessor):
#     """
#     NECO-style projection strength (ID-high):
#       - Collect ID features, L2-normalize, fit PCA on them;
#       - Score = || PCA(f) ||_2  (larger => more ID-like).
#     """

#     def __init__(self, config, **kwargs):
#         super().__init__(config)
#         pp = getattr(config, "postprocessor", None)

#         def _get(key, default):
#             try:
#                 if pp is None: return default
#                 if hasattr(pp, key): return getattr(pp, key)
#                 if hasattr(pp, "args") and hasattr(pp.args, key): return getattr(pp.args, key)
#                 if isinstance(pp, dict):
#                     if key in pp: return pp[key]
#                     if "args" in pp and key in pp["args"]: return pp["args"][key]
#             except Exception:
#                 pass
#             return default

#         self.n_components = _get("n_components", None)
#         self.eps = float(_get("eps", 1e-12))

#         self._hook_handle = None
#         self._pca = None

#     @staticmethod
#     def _find_feat_layer(net: nn.Module) -> nn.Module:
#         for name in ("avgpool", "global_pool", "pool", "head"):
#             if hasattr(net, name):
#                 return getattr(net, name)
#         children = list(net.children())
#         return children[-2] if len(children) >= 2 else net

#     @torch.no_grad()
#     def setup(self, net: nn.Module, id_loader: Any, ood_loader_dict: Any = None) -> None:
#         feat_layer = self._find_feat_layer(net)
#         if self._hook_handle is None:
#             self._hook_handle = feat_layer.register_forward_hook(feature_hook)

#         if isinstance(id_loader, dict):
#             id_loader = id_loader.get("train", next(iter(id_loader.values())))

#         device = next(net.parameters()).device
#         was_training = net.training
#         net.eval()

#         feats = []
#         for batch in id_loader:
#             data = batch["data"] if isinstance(batch, dict) else batch[0]
#             data = data.to(device, non_blocking=True)
#             _ = net(data)
#             if FeatureBank.value is None:
#                 raise RuntimeError("[neco] FeatureBank empty; hook not triggered.")
#             f = FeatureBank.value.detach().view(data.size(0), -1)
#             f = F.normalize(f, p=2, dim=1, eps=self.eps)   # L2-normalize
#             feats.append(f.cpu())

#         if was_training:
#             net.train()

#         all_feats = torch.cat(feats, dim=0).numpy()  # [N, D]
#         D = all_feats.shape[1]
#         k = self.n_components if self.n_components is not None else D
#         self._pca = PCA(n_components=min(k, D), svd_solver="auto")
#         self._pca.fit(all_feats)

#     @torch.no_grad()
#     def postprocess(self, net: nn.Module, data: Any):
#         if self._pca is None:
#             raise RuntimeError("[neco] PCA not fitted; call setup() first.")

#         logits = net(data)
#         if FeatureBank.value is None:
#             raise RuntimeError("[neco] FeatureBank empty during postprocess.")

#         feat = FeatureBank.value.detach().view(data.size(0), -1)
#         feat = F.normalize(feat, p=2, dim=1, eps=self.eps)

#         # PCA projection in numpy, then convert back
#         h_np = feat.cpu().numpy()
#         h_pca = self._pca.transform(h_np)          # [B, K]
#         score = torch.from_numpy((h_pca**2).sum(axis=1) ** 0.5).to(feat.device).to(feat.dtype)
#         # ID-high: larger projection norm => more ID-like

#         pred = torch.argmax(logits, dim=1)
#         return pred, score

#     def teardown(self):
#         if self._hook_handle is not None:
#             try:
#                 self._hook_handle.remove()
#             except Exception:
#                 pass
#             self._hook_handle = None

#     def __del__(self):
#         self.teardown()

# openood/postprocessors/neco_postprocessor.py
from typing import Any, Optional, Tuple
import numpy as np  # 目前只为兼容保留
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA  # 使用增量 PCA

from .base_postprocessor import BasePostprocessor
from openood.evaluation_api.featurebank import feature_hook, FeatureBank


class NecoPostprocessor(BasePostprocessor):
    """
    NECO-style projection strength (ID-high):
      - Collect ID features, L2-normalize, fit PCA on them;
      - Score = || PCA(f) ||_2  (larger => more ID-like).
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        pp = getattr(config, "postprocessor", None)

        def _get(key, default):
            try:
                if pp is None:
                    return default
                if hasattr(pp, key):
                    return getattr(pp, key)
                if hasattr(pp, "args") and hasattr(pp.args, key):
                    return getattr(pp.args, key)
                if isinstance(pp, dict):
                    if key in pp:
                        return pp[key]
                    if "args" in pp and key in pp["args"]:
                        return pp["args"][key]
            except Exception:
                pass
            return default

        self.n_components = _get("n_components", None)
        self.eps = float(_get("eps", 1e-12))

        self._hook_handle = None
        self._pca = None

    @staticmethod
    def _find_feat_layer(net: nn.Module) -> nn.Module:
        for name in ("avgpool", "global_pool", "pool", "head"):
            if hasattr(net, name):
                return getattr(net, name)
        children = list(net.children())
        return children[-2] if len(children) >= 2 else net

    @torch.no_grad()
    def setup(self, net: nn.Module, id_loader: Any, ood_loader_dict: Any = None) -> None:
        import time
        feat_layer = self._find_feat_layer(net)
        if self._hook_handle is None:
            self._hook_handle = feat_layer.register_forward_hook(feature_hook)

        if isinstance(id_loader, dict):
            id_loader = id_loader.get("train", next(iter(id_loader.values())))

        device = next(net.parameters()).device
        was_training = net.training
        net.eval()

        # ===== 加速与进度：最小改动 =====
        ipca_batch = 8192         # 比 4096 再大一点，减少分块次数
        max_fit_samples = 50000   # 仅用前 50k ID 样本拟合 PCA
        print(f"[neco] IncrementalPCA: ipca_batch={ipca_batch}, max_fit_samples={max_fit_samples}")

        D = None
        fitted = 0
        t0 = time.time()

        for bi, batch in enumerate(id_loader):
            data = batch["data"] if isinstance(batch, dict) else batch[0]
            bsz = data.size(0)
            if fitted >= max_fit_samples:
                break  # 提前停止，够拟合了

            # 若最后一批会超过上限，只取需要的那部分
            take = min(bsz, max_fit_samples - fitted)
            if take < bsz:
                data = data[:take]

            data = data.to(device, non_blocking=True)
            _ = net(data)
            if FeatureBank.value is None:
                raise RuntimeError("[neco] FeatureBank empty; hook not triggered.")
            f = FeatureBank.value.detach().view(data.size(0), -1)  # [B, D]
            f = F.normalize(f, p=2, dim=1, eps=self.eps)
            f_np = f.cpu().numpy()

            if D is None:
                D = f_np.shape[1]
                # IPCA 第一次 partial_fit 要求 n_components <= 首批样本数
                k = self.n_components if self.n_components is not None else D
                k = min(k, D, f_np.shape[0])
                self._pca = IncrementalPCA(n_components=k)
                print(f"[neco] D={D}, n_components={k}")

            # 分块 partial_fit
            start = 0
            N = f_np.shape[0]
            while start < N:
                end = min(start + ipca_batch, N)
                self._pca.partial_fit(f_np[start:end])
                start = end

            fitted += f_np.shape[0]
            if fitted % 10000 == 0 or bi % 50 == 0:
                elapsed = time.time() - t0
                print(f"[neco] Fitted {fitted}/{max_fit_samples} samples in {elapsed:.1f}s")

        print(f"[neco] IPCA fit done. total_used={fitted}, elapsed={time.time()-t0:.1f}s")

        if was_training:
            net.train()

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        if self._pca is None:
            raise RuntimeError("[neco] PCA not fitted; call setup() first.")

        logits = net(data)
        if FeatureBank.value is None:
            raise RuntimeError("[neco] FeatureBank empty during postprocess.")

        feat = FeatureBank.value.detach().view(data.size(0), -1)
        feat = F.normalize(feat, p=2, dim=1, eps=self.eps)

        # PCA projection in numpy, then convert back
        h_np = feat.cpu().numpy()
        h_pca = self._pca.transform(h_np)          # [B, K]
        score = torch.from_numpy((h_pca ** 2).sum(axis=1) ** 0.5).to(feat.device).to(feat.dtype)
        # ID-high: larger projection norm => more ID-like

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
