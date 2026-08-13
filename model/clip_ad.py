import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MODEL
from .modules.adapters import (
    CSSDPatchAdapter,
    LocalGlobalPatchAdapter,
    MambaResponseContext,
    MultiLayerLocalPatchAdapter,
    ResidualPatchAdapter,
)
from .modules.prompt_templates import (
    DEFAULT_ABNORMAL_TEMPLATES,
    DEFAULT_NORMAL_TEMPLATES,
    instantiate_templates,
)
from .modules.patch_features import fuse_layer_scores, project_clip_patch_tokens
from .modules.scoring import abnormal_minus_normal, mean_topk_score, upsample_patch_map


def _load_aaclip_vendor():
    package_name = "_aaclip_vendor"
    vendor_dir = Path(__file__).resolve().parents[1] / "third_party" / "AA-CLIP-main" / "model"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(vendor_dir)]
        sys.modules[package_name] = package
    clip_mod = importlib.import_module(f"{package_name}.clip")
    tokenizer_mod = importlib.import_module(f"{package_name}.tokenizer")
    return clip_mod, tokenizer_mod.tokenize


@MODEL.register_module
class CLIPNormalityAD(nn.Module):
    """Staged CLIP text anomaly baseline.

    This module intentionally keeps CLIP image/text encoders frozen and uses
    fixed prompt prototypes only. It outputs a global CLIP abnormality score,
    patch text anomaly map(s), an upsampled pixel map, and the image score
    S_global + beta * MeanTopK(A_pixel).
    """

    def __init__(
        self,
        model_name="ViT-L-14-336",
        img_size=336,
        pretrained="openai",
        clip_weights_path="",
        require_pretrained=True,
        stage="text_only",
        patch_layer=None,
        levels=None,
        multi_level_fusion="mean",
        layer_weights=None,
        adapter_type="none",
        adapter_kwargs=None,
        adapter_semantic="patch_mean",
        use_mamba_context=False,
        mamba_context_kwargs=None,
        use_arcc=False,
        arcc_kwargs=None,
        loss_normal_topk_weight=1.0,
        loss_consistency_weight=0.1,
        loss_image_normal_weight=0.1,
        loss_arcc_normal_weight=0.0,
        loss_arcc_cal_weight=0.0,
        use_supervised_masks=False,
        supervised_mask_bce_weight=1.0,
        supervised_mask_dice_weight=1.0,
        supervised_raw_bce_weight=0.2,
        supervised_image_weight=0.1,
        supervised_outside_topk_weight=0.0,
        supervised_outside_topk_ratio=0.01,
        supervised_score_temperature=1.0,
        mamba_context_bce_weight=0.0,
        mamba_context_dice_weight=0.0,
        mamba_context_outside_topk_weight=0.0,
        mamba_context_outside_topk_ratio=0.01,
        loss_topk_ratio=0.01,
        surgery_until_layer=None,
        normal_templates=None,
        abnormal_templates=None,
        image_score_topk_ratio=0.01,
        topk_beta=0.5,
        arcc_mamba_context_scale=0.1,
        global_score_weight=None,
        margin=0.2,
        **unused_kwargs,
    ):
        super().__init__()
        if stage not in (
            "text_only",
            "stage1",
            "patch",
            "global",
            "multi_level_text",
            "stage2",
            "patch_adapter",
            "stage2a",
            "stage2b",
        ):
            raise ValueError(f"Unsupported CLIPNormalityAD stage: {stage}")

        clip_mod, tokenize = _load_aaclip_vendor()
        self.tokenize = tokenize
        weights_path = Path(clip_weights_path) if clip_weights_path else None
        if weights_path is not None:
            clip_mod._MODEL_CKPT_PATHS[model_name] = weights_path
        elif pretrained and str(pretrained).lower() == "openai":
            weights_path = clip_mod._MODEL_CKPT_PATHS.get(model_name)
        if require_pretrained and weights_path is not None and not weights_path.exists():
            raise FileNotFoundError(
                f"CLIP weights not found: {weights_path}. "
                "Download ViT-L-14-336px.pt and place it under "
                "third_party/AA-CLIP-main/model/, or set model.kwargs.clip_weights_path."
            )

        self.clip_model = clip_mod.create_model(
            model_name=model_name,
            img_size=img_size,
            pretrained=pretrained,
            require_pretrained=require_pretrained,
            device="cpu",
        )
        self.clip_model.eval()
        if surgery_until_layer is not None and hasattr(self.clip_model.visual, "DAPM_replace"):
            self.clip_model.visual.DAPM_replace(DPAM_layer=surgery_until_layer)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        num_layers = len(self.clip_model.visual.transformer.resblocks)
        self.adapter_type = str(adapter_type).lower()
        if stage == "stage2a" and self.adapter_type == "none":
            self.adapter_type = "mlp"
        if stage == "stage2b" and self.adapter_type == "none":
            self.adapter_type = "cssd"
        if self.adapter_type not in ("none", "mlp", "cssd", "local_global", "multilayer_local"):
            raise ValueError(
                f"Unsupported adapter_type={adapter_type}. Expected none, mlp, cssd, local_global, or multilayer_local."
            )
        self.use_patch_adapter = self.adapter_type != "none"
        self.use_multi_level = stage in ("multi_level_text", "stage2") and not self.use_patch_adapter
        self.use_arcc = bool(use_arcc)
        if self.adapter_type == "multilayer_local":
            if levels:
                out_layers = list(levels)
            else:
                final_layer = patch_layer or num_layers
                out_layers = [12, 18, int(final_layer)]
        elif self.use_multi_level:
            out_layers = list(levels or (6, 12, 18, num_layers))
        else:
            if patch_layer is None:
                if levels:
                    patch_layer = list(levels)[-1]
                else:
                    patch_layer = num_layers
            out_layers = [int(patch_layer)]
        out_layers = [int(layer) for layer in out_layers]
        for layer in out_layers:
            if layer < 1 or layer > num_layers:
                raise ValueError(f"patch layer must be in [1, {num_layers}], got {layer}.")
        if len(set(out_layers)) != len(out_layers):
            raise ValueError(f"Duplicate patch layers are not supported: {out_layers}.")
        self.out_layers = out_layers
        self.img_size = img_size
        self.use_mamba_context = bool(use_mamba_context)
        if self.use_patch_adapter:
            self.stage = f"{self.adapter_type}_adapter_text"
        else:
            self.stage = "multi_level_text" if self.use_multi_level else "text_only"
        if self.use_mamba_context:
            self.stage = f"{self.stage}_lastpatch_mamba"
        if self.use_arcc:
            self.stage = f"{self.stage}_arcc"
        self.multi_level_fusion = multi_level_fusion
        self.layer_weights = layer_weights
        self.adapter_semantic = str(adapter_semantic).lower()
        self.loss_normal_topk_weight = float(loss_normal_topk_weight)
        self.loss_consistency_weight = float(loss_consistency_weight)
        self.loss_image_normal_weight = float(loss_image_normal_weight)
        self.loss_arcc_normal_weight = float(loss_arcc_normal_weight)
        self.loss_arcc_cal_weight = float(loss_arcc_cal_weight)
        self.use_supervised_masks = bool(use_supervised_masks)
        self.supervised_mask_bce_weight = float(supervised_mask_bce_weight)
        self.supervised_mask_dice_weight = float(supervised_mask_dice_weight)
        self.supervised_raw_bce_weight = float(supervised_raw_bce_weight)
        self.supervised_image_weight = float(supervised_image_weight)
        self.supervised_outside_topk_weight = float(supervised_outside_topk_weight)
        self.supervised_outside_topk_ratio = supervised_outside_topk_ratio
        self.supervised_score_temperature = max(float(supervised_score_temperature), 1e-6)
        self.mamba_context_bce_weight = float(mamba_context_bce_weight)
        self.mamba_context_dice_weight = float(mamba_context_dice_weight)
        self.mamba_context_outside_topk_weight = float(mamba_context_outside_topk_weight)
        self.mamba_context_outside_topk_ratio = mamba_context_outside_topk_ratio
        self.loss_topk_ratio = loss_topk_ratio
        self.margin = margin
        self.image_score_topk_ratio = image_score_topk_ratio
        self.topk_beta = float(topk_beta)
        self.arcc_mamba_context_scale = float(arcc_mamba_context_scale)
        if global_score_weight is not None:
            # Backward-compatible alias for older weighted-fusion configs.
            self.topk_beta = float(1.0 - global_score_weight)
        if self.image_score_topk_ratio is None:
            self.stage = f"{self.stage}_scoremax"
        if self.supervised_outside_topk_weight > 0:
            suffix = str(self.supervised_outside_topk_weight).replace(".", "p")
            self.stage = f"{self.stage}_outside{suffix}"

        self.normal_templates = list(normal_templates or DEFAULT_NORMAL_TEMPLATES)
        self.abnormal_templates = list(abnormal_templates or DEFAULT_ABNORMAL_TEMPLATES)
        self._text_cache = {}

        embed_dim = int(self.clip_model.text_projection.shape[1])
        adapter_kwargs = dict(adapter_kwargs or {})
        self.patch_adapter = None
        if self.adapter_type == "mlp":
            self.patch_adapter = ResidualPatchAdapter(embed_dim, **adapter_kwargs)
        elif self.adapter_type == "cssd":
            grid_h, grid_w = self._grid_size()
            if grid_h != grid_w:
                raise ValueError(f"CSSD adapter requires square CLIP patch grid, got {(grid_h, grid_w)}.")
            self.patch_adapter = CSSDPatchAdapter(embed_dim, grid_size=grid_h, **adapter_kwargs)
        elif self.adapter_type == "local_global":
            grid_h, grid_w = self._grid_size()
            if grid_h != grid_w:
                raise ValueError(f"Local-global adapter requires square CLIP patch grid, got {(grid_h, grid_w)}.")
            self.patch_adapter = LocalGlobalPatchAdapter(embed_dim, grid_size=grid_h, **adapter_kwargs)
        elif self.adapter_type == "multilayer_local":
            grid_h, grid_w = self._grid_size()
            if grid_h != grid_w:
                raise ValueError(f"Multi-layer local adapter requires square CLIP patch grid, got {(grid_h, grid_w)}.")
            local_layers = int(adapter_kwargs.pop("local_layers", max(1, len(self.out_layers) - 1)))
            self.patch_adapter = MultiLayerLocalPatchAdapter(
                embed_dim,
                grid_size=grid_h,
                local_layers=local_layers,
                **adapter_kwargs,
            )

        self.mamba_context = None
        if self.use_mamba_context:
            grid_h, grid_w = self._grid_size()
            if grid_h != grid_w:
                raise ValueError(f"Mamba response context requires square CLIP patch grid, got {(grid_h, grid_w)}.")
            self.mamba_context = MambaResponseContext(
                embed_dim,
                grid_size=grid_h,
                **dict(mamba_context_kwargs or {}),
            )

        self.arcc = None
        self.arcc_lambda = None
        if self.use_arcc:
            from model.mambaad import ARCCCalibration

            arcc_kwargs = dict(arcc_kwargs or {})
            self.arcc_lambda = nn.Parameter(torch.tensor(float(arcc_kwargs.get("lambda_init", 0.1))))
            self.arcc = ARCCCalibration(
                embed_dim,
                use_response=bool(arcc_kwargs.get("use_response", True)),
                use_foreground=bool(arcc_kwargs.get("use_foreground", False)),
                use_edge=bool(arcc_kwargs.get("use_edge", False)),
                kernel_size=int(arcc_kwargs.get("kernel_size", 3)),
                hidden_dim=arcc_kwargs.get("hidden_dim", None),
                lambda_init=float(arcc_kwargs.get("lambda_init", 0.1)),
            )

        # Keeps the legacy training loop/optimizer usable while preserving a
        # true frozen-CLIP baseline. The parameter is multiplied by zero.
        self.stage1_anchor = nn.Parameter(torch.zeros(()))

    def train(self, mode=True):
        self.training = mode
        self.clip_model.eval()
        if self.patch_adapter is not None:
            self.patch_adapter.train(mode)
        if self.mamba_context is not None:
            self.mamba_context.train(mode)
        if self.arcc is not None:
            self.arcc.train(mode)
        return self

    def _encode_prompt_group(self, prompts, device):
        tokens = self.tokenize(prompts).to(device)
        with torch.no_grad():
            feats = self.clip_model.encode_text(tokens)
            feats = F.normalize(feats.float(), dim=-1)
        return F.normalize(feats.mean(dim=0), dim=0)

    def _fixed_text_prototypes(self, cls_name, device):
        key = (str(cls_name), str(device))
        cached = self._text_cache.get(key)
        if cached is None:
            normal_prompts = instantiate_templates(self.normal_templates, cls_name)
            abnormal_prompts = instantiate_templates(self.abnormal_templates, cls_name)
            normal = self._encode_prompt_group(normal_prompts, device)
            abnormal = self._encode_prompt_group(abnormal_prompts, device)
            cached = torch.stack([normal, abnormal], dim=0)
            self._text_cache[key] = cached
        return cached

    def text_prototypes(self, cls_names, device):
        if isinstance(cls_names, str):
            cls_names = [cls_names]
        protos = [self._fixed_text_prototypes(cls_name, device) for cls_name in cls_names]
        return torch.stack(protos, dim=0)

    def _grid_size(self):
        grid = self.clip_model.visual.grid_size
        if isinstance(grid, tuple):
            return int(grid[0]), int(grid[1])
        return int(grid), int(grid)

    def _num_tokens_with_cls(self):
        h, w = self._grid_size()
        return h * w + 1

    def encode_image_features(self, imgs):
        with torch.no_grad():
            global_feat, patch_layers = self.clip_model.encode_image(imgs, self.out_layers)
            global_feat = F.normalize(global_feat.float(), dim=-1)
            if not patch_layers:
                raise RuntimeError(f"No patch tokens returned for out_layers={self.out_layers}.")
            patch_feats = [
                project_clip_patch_tokens(
                    self.clip_model.visual,
                    patch_tokens,
                    self._num_tokens_with_cls(),
                )
                for patch_tokens in patch_layers
            ]
        return global_feat, patch_feats

    def _semantic_embedding(self, patch_feat, protos):
        if self.adapter_semantic in ("patch", "patch_mean", "visual"):
            return F.normalize(patch_feat.mean(dim=1), dim=-1)
        normal = protos[:, 0]
        abnormal = protos[:, 1]
        if self.adapter_semantic == "normal":
            return normal
        if self.adapter_semantic == "mean":
            return F.normalize(0.5 * (normal + abnormal), dim=-1)
        if self.adapter_semantic == "abnormal":
            return abnormal
        raise ValueError(f"Unsupported adapter_semantic={self.adapter_semantic}.")

    def _apply_patch_adapter(self, patch_feat, protos, patch_feats=None):
        if self.patch_adapter is None:
            return patch_feat, None
        spatial_shape = self._grid_size()
        if self.adapter_type == "multilayer_local":
            if patch_feats is None or len(patch_feats) < 2:
                raise ValueError("multilayer_local adapter requires low/mid patch features plus the final patch feature.")
            local_patches = patch_feats[:-1]
            return self.patch_adapter(patch_feat, local_patches, spatial_shape)
        semantic_embedding = self._semantic_embedding(patch_feat, protos)
        return self.patch_adapter(patch_feat, semantic_embedding, spatial_shape), None

    def _patch_topk(self, patch_score):
        if self.loss_topk_ratio is None:
            return patch_score.max(dim=1).values
        topk = max(1, int(patch_score.shape[1] * float(self.loss_topk_ratio)))
        return patch_score.topk(topk, dim=1).values.mean(dim=1)

    def _tokens_to_feature_map(self, tokens):
        grid_h, grid_w = self._grid_size()
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], grid_h, grid_w)

    def _apply_arcc(self, tokens, raw_patch_map, raw_anomaly_map, image_shape, mamba_source_tokens=None):
        if self.arcc is None:
            return raw_anomaly_map, None, tokens
        context_tokens = tokens
        mamba_debug = {}
        feature_map = self._tokens_to_feature_map(tokens)
        if self.mamba_context is not None:
            mamba_source_tokens = tokens if mamba_source_tokens is None else mamba_source_tokens
            mamba_tokens, mamba_debug = self.mamba_context(
                mamba_source_tokens,
                self._grid_size(),
            )
            mamba_feature_map = self._tokens_to_feature_map(mamba_tokens)
            feature_map = feature_map + self.arcc_mamba_context_scale * mamba_feature_map
            context_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        g_cal, mod_mask = self.arcc(
            feature_map,
            raw_patch_map,
            foreground=None,
            edge=None,
            image_shape=image_shape,
        )
        arcc_lambda = torch.clamp(self.arcc_lambda, min=0.0, max=2.0)
        final_map = raw_anomaly_map + arcc_lambda * raw_anomaly_map * torch.tanh(g_cal)
        return final_map, {
            "G_cal": g_cal,
            "arcc_mod_mask": mod_mask,
            "arcc_lambda": arcc_lambda,
            "mamba_context_tokens": context_tokens,
            **mamba_debug,
        }, context_tokens

    def _dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        intersection = probs * targets
        dice = (2.0 * intersection.sum(dim=1) + 1.0) / (
            probs.sum(dim=1) + targets.sum(dim=1) + 1.0
        )
        return 1.0 - dice.mean()

    def _supervised_mask_losses(self, anomaly_map, raw_anomaly_map, masks):
        targets = masks.to(device=anomaly_map.device, dtype=anomaly_map.dtype)
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.shape[-2:] != anomaly_map.shape[-2:]:
            targets = F.interpolate(targets, size=anomaly_map.shape[-2:], mode="nearest")
        targets = targets.clamp(0.0, 1.0)
        logits = anomaly_map.unsqueeze(1) / self.supervised_score_temperature
        raw_logits = raw_anomaly_map.unsqueeze(1) / self.supervised_score_temperature
        loss_mask_bce = F.binary_cross_entropy_with_logits(logits, targets)
        loss_mask_dice = self._dice_loss(logits, targets)
        loss_mask_raw_bce = F.binary_cross_entropy_with_logits(raw_logits, targets)
        return loss_mask_bce, loss_mask_dice, loss_mask_raw_bce

    def _outside_topk_loss_with_ratio(self, anomaly_map, masks, topk_ratio):
        targets = masks.to(device=anomaly_map.device, dtype=anomaly_map.dtype)
        if targets.ndim == 4:
            targets = targets.squeeze(1)
        if targets.shape[-2:] != anomaly_map.shape[-2:]:
            targets = F.interpolate(
                targets.unsqueeze(1),
                size=anomaly_map.shape[-2:],
                mode="nearest",
            ).squeeze(1)
        outside = targets <= 0.5
        losses = []
        flat_scores = anomaly_map.flatten(1)
        flat_outside = outside.flatten(1)
        for scores, outside_mask in zip(flat_scores, flat_outside):
            selected = scores[outside_mask]
            if selected.numel() == 0:
                continue
            if topk_ratio is None:
                selected = selected.max().view(1)
            else:
                topk = max(1, int(selected.numel() * float(topk_ratio)))
                selected = selected.topk(topk).values
            losses.append(F.softplus(selected + self.margin).mean())
        if not losses:
            return anomaly_map.new_zeros(())
        return torch.stack(losses).mean()

    def _outside_topk_loss(self, anomaly_map, masks):
        return self._outside_topk_loss_with_ratio(
            anomaly_map,
            masks,
            self.supervised_outside_topk_ratio,
        )

    def _mamba_context_losses(self, context_logits, masks):
        targets = masks.to(device=context_logits.device, dtype=context_logits.dtype)
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.shape[-2:] != context_logits.shape[-2:]:
            targets = F.interpolate(targets, size=context_logits.shape[-2:], mode="nearest")
        targets = targets.clamp(0.0, 1.0)
        logits = context_logits.unsqueeze(1) / self.supervised_score_temperature
        loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
        loss_dice = self._dice_loss(logits, targets)
        loss_outside = self._outside_topk_loss_with_ratio(
            context_logits,
            targets,
            self.mamba_context_outside_topk_ratio,
        )
        return loss_bce, loss_dice, loss_outside

    def _adapter_losses(
        self,
        raw_patch,
        refined_patch,
        patch_score,
        image_score,
        anomaly_map=None,
        raw_anomaly_map=None,
        arcc_debug=None,
        masks=None,
        labels=None,
    ):
        normal_patch_score = patch_score
        if labels is not None:
            normal_mask = labels.to(device=patch_score.device).view(-1) == 0
            normal_patch_score = patch_score[normal_mask]
        if normal_patch_score.numel() > 0:
            selected = self._patch_topk(normal_patch_score)
            loss_normal_topk = F.softplus(selected + self.margin).mean()
        else:
            loss_normal_topk = image_score.new_zeros(())
        raw_norm = F.normalize(raw_patch.detach(), dim=-1)
        refined_norm = F.normalize(refined_patch, dim=-1)
        loss_consistency = 1.0 - torch.sum(refined_norm * raw_norm, dim=-1).mean()
        image_target = torch.zeros_like(image_score)
        loss_image_supervised = image_score.new_zeros(())
        if self.use_supervised_masks and labels is not None:
            image_target = labels.to(device=image_score.device, dtype=image_score.dtype).view_as(image_score)
            loss_image_supervised = F.binary_cross_entropy_with_logits(image_score, image_target)
        loss_image_normal = F.binary_cross_entropy_with_logits(image_score, image_target)
        loss_arcc_normal = image_score.new_zeros(())
        loss_arcc_cal = image_score.new_zeros(())
        if self.arcc is not None and anomaly_map is not None and arcc_debug is not None:
            arcc_target = torch.zeros_like(anomaly_map)
            if self.use_supervised_masks and masks is not None:
                arcc_target = masks.to(device=anomaly_map.device, dtype=anomaly_map.dtype)
                if arcc_target.ndim == 4:
                    arcc_target = arcc_target.squeeze(1)
                if arcc_target.shape[-2:] != anomaly_map.shape[-2:]:
                    arcc_target = F.interpolate(
                        arcc_target.unsqueeze(1), size=anomaly_map.shape[-2:], mode="nearest"
                    ).squeeze(1)
            loss_arcc_normal = F.binary_cross_entropy_with_logits(anomaly_map, arcc_target)
            loss_arcc_cal = arcc_debug["G_cal"].pow(2).mean()
        loss_mask_bce = image_score.new_zeros(())
        loss_mask_dice = image_score.new_zeros(())
        loss_mask_raw_bce = image_score.new_zeros(())
        loss_outside_topk = image_score.new_zeros(())
        loss_mamba_context_bce = image_score.new_zeros(())
        loss_mamba_context_dice = image_score.new_zeros(())
        loss_mamba_context_outside_topk = image_score.new_zeros(())
        if self.use_supervised_masks and masks is not None and anomaly_map is not None and raw_anomaly_map is not None:
            loss_mask_bce, loss_mask_dice, loss_mask_raw_bce = self._supervised_mask_losses(
                anomaly_map,
                raw_anomaly_map,
                masks,
            )
            if self.supervised_outside_topk_weight > 0:
                loss_outside_topk = self._outside_topk_loss(anomaly_map, masks)
        if (
            self.use_supervised_masks
            and masks is not None
            and arcc_debug is not None
            and "mamba_context_logits" in arcc_debug
            and (
                self.mamba_context_bce_weight > 0
                or self.mamba_context_dice_weight > 0
                or self.mamba_context_outside_topk_weight > 0
            )
        ):
            (
                loss_mamba_context_bce,
                loss_mamba_context_dice,
                loss_mamba_context_outside_topk,
            ) = self._mamba_context_losses(arcc_debug["mamba_context_logits"], masks)
        total = (
            self.loss_normal_topk_weight * loss_normal_topk
            + self.loss_consistency_weight * loss_consistency
            + self.loss_image_normal_weight * loss_image_normal
            + self.loss_arcc_normal_weight * loss_arcc_normal
            + self.loss_arcc_cal_weight * loss_arcc_cal
            + self.supervised_mask_bce_weight * loss_mask_bce
            + self.supervised_mask_dice_weight * loss_mask_dice
            + self.supervised_raw_bce_weight * loss_mask_raw_bce
            + self.supervised_outside_topk_weight * loss_outside_topk
            + self.supervised_image_weight * loss_image_supervised
            + self.mamba_context_bce_weight * loss_mamba_context_bce
            + self.mamba_context_dice_weight * loss_mamba_context_dice
            + self.mamba_context_outside_topk_weight * loss_mamba_context_outside_topk
        )
        return {
            "loss_normal_topk": loss_normal_topk,
            "loss_consistency": loss_consistency,
            "loss_image_normal": loss_image_normal,
            "loss_arcc_normal": loss_arcc_normal,
            "loss_arcc_cal": loss_arcc_cal,
            "loss_mask_bce": loss_mask_bce,
            "loss_mask_dice": loss_mask_dice,
            "loss_mask_raw_bce": loss_mask_raw_bce,
            "loss_outside_topk": loss_outside_topk,
            "loss_image_supervised": loss_image_supervised,
            "loss_mamba_context_bce": loss_mamba_context_bce,
            "loss_mamba_context_dice": loss_mamba_context_dice,
            "loss_mamba_context_outside_topk": loss_mamba_context_outside_topk,
            "loss_patch": loss_normal_topk,
            "total": total,
            "loss_total": total,
        }

    def _debug_scalars(
        self,
        patch_feats,
        raw_patch_feat,
        refined_patch_feat,
        context_patch_feat,
        local_delta,
        s_global,
        s_text_patch,
        raw_anomaly_map,
        anomaly_map,
        arcc_debug,
        topk_score,
        image_score_variants,
        masks=None,
        labels=None,
    ):
        with torch.no_grad():
            eps = 1e-6
            raw_norm = F.normalize(raw_patch_feat.detach(), dim=-1)
            refined_norm = F.normalize(refined_patch_feat.detach(), dim=-1)
            context_norm = F.normalize(context_patch_feat.detach(), dim=-1)
            refine_delta = refined_patch_feat.detach() - raw_patch_feat.detach()
            context_delta = context_patch_feat.detach() - refined_patch_feat.detach()
            debug = {
                "dbg_refine_cos": torch.sum(raw_norm * refined_norm, dim=-1).mean(),
                "dbg_refine_delta_l2": refine_delta.float().pow(2).sum(dim=-1).sqrt().mean(),
                "dbg_mamba_context_cos": torch.sum(refined_norm * context_norm, dim=-1).mean(),
                "dbg_mamba_context_delta_l2": context_delta.float().pow(2).sum(dim=-1).sqrt().mean(),
                "dbg_a_raw_mean": raw_anomaly_map.detach().mean(),
                "dbg_a_raw_max": raw_anomaly_map.detach().amax(),
                "dbg_a_final_mean": anomaly_map.detach().mean(),
                "dbg_a_final_max": anomaly_map.detach().amax(),
                "dbg_s_global": s_global.detach().mean(),
                "dbg_topk_score": topk_score.detach().mean(),
                "dbg_topk_score_max": image_score_variants["topk_score_max"].detach().mean(),
                "dbg_topk_score_top1": image_score_variants["topk_score_top1"].detach().mean(),
                "dbg_topk_score_top5": image_score_variants["topk_score_top5"].detach().mean(),
                "dbg_image_score_max": image_score_variants["image_score_max"].detach().mean(),
                "dbg_image_score_top1": image_score_variants["image_score_top1"].detach().mean(),
                "dbg_image_score_top5": image_score_variants["image_score_top5"].detach().mean(),
            }
            if local_delta is not None:
                local = local_delta.detach().float()
                debug.update(
                    {
                        "dbg_local_delta_l2": local.pow(2).sum(dim=-1).sqrt().mean(),
                        "dbg_local_delta_abs": local.abs().mean(),
                    }
                )
            if len(patch_feats) >= 2:
                last = F.normalize(patch_feats[-1].detach(), dim=-1)
                for idx, patch_feat in enumerate(patch_feats[:-1]):
                    layer = self.out_layers[idx] if idx < len(self.out_layers) else idx
                    patch_norm = F.normalize(patch_feat.detach(), dim=-1)
                    debug[f"dbg_l{layer}_last_cos"] = torch.sum(patch_norm * last, dim=-1).mean()
            arcc_delta = anomaly_map.detach() - raw_anomaly_map.detach()
            debug["dbg_arcc_delta_abs"] = arcc_delta.abs().mean()
            debug["dbg_arcc_delta_ratio"] = arcc_delta.abs().mean() / raw_anomaly_map.detach().abs().mean().clamp_min(eps)
            if arcc_debug is not None:
                debug["dbg_arcc_lambda"] = arcc_debug["arcc_lambda"].detach()
                debug["dbg_g_cal_abs"] = arcc_debug["G_cal"].detach().abs().mean()
                if "mamba_global_prior" in arcc_debug:
                    prior = arcc_debug["mamba_global_prior"].detach()
                    debug["dbg_mamba_prior_mean"] = prior.mean()
                    debug["dbg_mamba_prior_max"] = prior.amax()
                    prior_topk = prior.flatten(1).topk(
                        max(1, int(prior[0].numel() * 0.01)), dim=1
                    ).values.mean(dim=1)
                    if labels is not None:
                        label_values = labels.to(device=prior.device).view(-1)
                        normal = prior_topk[label_values == 0]
                        abnormal = prior_topk[label_values != 0]
                        if normal.numel() > 0:
                            debug["dbg_mamba_prior_normal_topk"] = normal.mean()
                        if abnormal.numel() > 0:
                            debug["dbg_mamba_prior_abnormal_topk"] = abnormal.mean()
                    if masks is not None:
                        target = masks.to(device=prior.device, dtype=prior.dtype)
                        if target.ndim == 4:
                            target = target.squeeze(1)
                        if target.shape[-2:] != prior.shape[-2:]:
                            target = F.interpolate(
                                target.unsqueeze(1), size=prior.shape[-2:], mode="nearest"
                            ).squeeze(1)
                        inside = prior[target > 0.5]
                        outside = prior[target <= 0.5]
                        inside_mean = inside.mean() if inside.numel() > 0 else prior.new_zeros(())
                        outside_mean = outside.mean() if outside.numel() > 0 else prior.new_zeros(())
                        debug["dbg_mamba_prior_mask_in"] = inside_mean
                        debug["dbg_mamba_prior_mask_out"] = outside_mean
                        debug["dbg_mamba_prior_gap"] = inside_mean - outside_mean
                if "mamba_depth_weight_mean" in arcc_debug:
                    for idx, weight in enumerate(arcc_debug["mamba_depth_weight_mean"].detach()):
                        debug[f"dbg_mamba_depth_w_f{idx}"] = weight
            else:
                debug["dbg_arcc_lambda"] = raw_anomaly_map.new_zeros(())
                debug["dbg_g_cal_abs"] = raw_anomaly_map.new_zeros(())
            return debug

    def forward(self, imgs, cls_names=None, masks=None, labels=None, return_loss=True):
        if cls_names is None:
            cls_names = ["object"] * imgs.shape[0]
        global_feat, patch_feats = self.encode_image_features(imgs)
        protos = self.text_prototypes(cls_names, imgs.device)

        s_global, global_sim = abnormal_minus_normal(global_feat, protos)
        raw_patch_feat = patch_feats[-1]
        refined_patch_feat, local_delta = self._apply_patch_adapter(raw_patch_feat, protos, patch_feats)

        if self.use_patch_adapter:
            s_text_patch, patch_sim = abnormal_minus_normal(refined_patch_feat, protos)
            s_text_map = s_text_patch.reshape(s_text_patch.shape[0], *self._grid_size())
            layer_text_maps = [s_text_map]
        elif self.use_multi_level:
            layer_patch_scores = []
            layer_patch_sims = []
            layer_text_maps = []
            for patch_feat in patch_feats:
                patch_score, patch_sim = abnormal_minus_normal(patch_feat, protos)
                layer_patch_scores.append(patch_score)
                layer_patch_sims.append(patch_sim)
                layer_text_maps.append(patch_score.reshape(patch_score.shape[0], *self._grid_size()))
            s_text_patch = fuse_layer_scores(
                layer_patch_scores,
                mode=self.multi_level_fusion,
                weights=self.layer_weights,
            )
            patch_sim = layer_patch_sims[-1]
            s_text_map = s_text_patch.reshape(s_text_patch.shape[0], *self._grid_size())
        else:
            s_text_patch, patch_sim = abnormal_minus_normal(refined_patch_feat, protos)
            s_text_map = s_text_patch.reshape(s_text_patch.shape[0], *self._grid_size())
            layer_text_maps = [s_text_map]
        raw_anomaly_map = upsample_patch_map(
            s_text_patch,
            grid_size=self._grid_size(),
            image_shape=(imgs.shape[2], imgs.shape[3]),
        )
        anomaly_map, arcc_debug, context_patch_feat = self._apply_arcc(
            refined_patch_feat,
            s_text_map,
            raw_anomaly_map,
            image_shape=(imgs.shape[2], imgs.shape[3]),
            mamba_source_tokens=raw_patch_feat,
        )
        topk_score = mean_topk_score(anomaly_map, self.image_score_topk_ratio)
        image_score = s_global + self.topk_beta * topk_score
        topk_score_max = mean_topk_score(anomaly_map, None)
        topk_score_top1 = mean_topk_score(anomaly_map, 0.01)
        topk_score_top5 = mean_topk_score(anomaly_map, 0.05)
        image_score_variants = {
            "topk_score_max": topk_score_max,
            "topk_score_top1": topk_score_top1,
            "topk_score_top5": topk_score_top5,
            "image_score_max": s_global + self.topk_beta * topk_score_max,
            "image_score_top1": s_global + self.topk_beta * topk_score_top1,
            "image_score_top5": s_global + self.topk_beta * topk_score_top5,
        }

        out = {
            "S_global": s_global,
            "S_text_map": s_text_map,
            "A_pixel": anomaly_map,
            "A_raw": raw_anomaly_map,
            "A_final": anomaly_map,
            "topk_score": topk_score,
            "image_score": image_score,
            **image_score_variants,
            "anomaly_map": anomaly_map,
            "global_sim": global_sim,
            "patch_sim": patch_sim,
            "patch_layers": self.out_layers,
            "layer_text_maps": torch.stack(layer_text_maps, dim=1),
            "raw_patch_feat": raw_patch_feat,
            "refined_patch_feat": refined_patch_feat,
            "context_patch_feat": context_patch_feat,
        }
        if local_delta is not None:
            out["local_delta"] = local_delta
        out.update(
            self._debug_scalars(
                patch_feats,
                raw_patch_feat,
                refined_patch_feat,
                context_patch_feat,
                local_delta,
                s_global,
                s_text_patch,
                raw_anomaly_map,
                anomaly_map,
                arcc_debug,
                topk_score,
                image_score_variants,
                masks=masks,
                labels=labels,
            )
        )
        if arcc_debug is not None:
            out.update(arcc_debug)
        if self.use_multi_level:
            out["S_text_map_multi"] = s_text_map
        if return_loss:
            loss_global = F.softplus(s_global + self.margin).mean()
            if self.use_patch_adapter:
                losses = self._adapter_losses(
                    raw_patch_feat,
                    refined_patch_feat,
                    s_text_patch,
                    image_score,
                    anomaly_map=anomaly_map,
                    raw_anomaly_map=raw_anomaly_map,
                    arcc_debug=arcc_debug,
                    masks=masks,
                    labels=labels,
                )
                total = losses["total"]
            else:
                loss_patch = F.softplus(s_text_patch + self.margin).mean()
                total = loss_global.detach() + loss_patch.detach() + self.stage1_anchor * 0.0
                losses = {
                    "loss_patch": loss_patch,
                    "total": total,
                    "loss_total": total,
                }
            out.update(
                {
                    "loss_global": loss_global,
                    **losses,
                }
            )
        return out
