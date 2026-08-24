import importlib
import math
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
from util.mamba_veto import apply_mamba_probability_veto, resize_mamba_patch_targets


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
        mamba_context_separation_weight=0.0,
        mamba_context_separation_margin=0.2,
        mamba_feature_contrast_target_weight=0.0,
        mamba_feature_contrast_warmup_epochs=0,
        mamba_feature_contrast_temperature=0.1,
        mamba_feature_contrast_hard_negative_ratio=0.05,
        mamba_context_mask_pool="nearest",
        loss_topk_ratio=0.01,
        surgery_until_layer=None,
        normal_templates=None,
        abnormal_templates=None,
        image_score_topk_ratio=0.01,
        topk_beta=0.5,
        image_score_mode="legacy",
        image_reviewer_raw_temperature=1.0,
        image_reviewer_topk_ratio=0.05,
        image_reviewer_weight_init=(1.0, 0.5, 0.5, 0.5),
        image_relative_reviewer_margin=0.01,
        image_relative_reviewer_max_scale=0.3,
        image_relative_reviewer_scale_init=0.05,
        image_relative_reviewer_base_weight_init=(1.0, 1e-4, 1e-4, 0.5),
        image_relative_reviewer_detach_mamba=True,
        pdar_image_pool_temperature=2.0,
        pdar_image_scale_init=0.1,
        pdar_image_dropout=0.1,
        pdar_image_attention_detach=True,
        pdar_image_loss_weight=0.0,
        arcc_mamba_context_scale=0.1,
        arcc_inject_mamba=False,
        arcc_mamba_injection_init=0.1,
        arcc_mamba_fusion_mode="add",
        arcc_mamba_feature_source="context_tokens",
        arcc_mamba_support_guidance=False,
        arcc_mamba_support_detach=True,
        arcc_lambda_override=None,
        arcc_variant="deformable",
        arcc_mode="bidirectional",
        mamba_veto_source="semantic",
        mamba_veto_alpha_init=0.1,
        mamba_veto_temperature=1.0,
        mamba_veto_threshold=0.0,
        mamba_veto_detach=True,
        mamba_veto_enabled=True,
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
        self.mamba_context_separation_weight = float(mamba_context_separation_weight)
        self.mamba_context_separation_margin = float(mamba_context_separation_margin)
        self.mamba_feature_contrast_target_weight = float(
            mamba_feature_contrast_target_weight
        )
        self.mamba_feature_contrast_warmup_epochs = int(
            mamba_feature_contrast_warmup_epochs
        )
        self.mamba_feature_contrast_temperature = float(
            mamba_feature_contrast_temperature
        )
        self.mamba_feature_contrast_hard_negative_ratio = float(
            mamba_feature_contrast_hard_negative_ratio
        )
        if self.mamba_feature_contrast_target_weight < 0.0:
            raise ValueError("mamba_feature_contrast_target_weight must be non-negative.")
        if self.mamba_feature_contrast_warmup_epochs < 0:
            raise ValueError("mamba_feature_contrast_warmup_epochs must be non-negative.")
        if self.mamba_feature_contrast_temperature <= 0.0:
            raise ValueError("mamba_feature_contrast_temperature must be positive.")
        if not 0.0 < self.mamba_feature_contrast_hard_negative_ratio <= 1.0:
            raise ValueError(
                "mamba_feature_contrast_hard_negative_ratio must be in (0, 1]."
            )
        self.mamba_context_mask_pool = str(mamba_context_mask_pool).lower()
        if self.mamba_context_mask_pool not in ("nearest", "adaptive_max"):
            raise ValueError(
                "mamba_context_mask_pool must be 'nearest' or 'adaptive_max', "
                f"got {mamba_context_mask_pool!r}."
            )
        self.loss_topk_ratio = loss_topk_ratio
        self.margin = margin
        self.image_score_topk_ratio = image_score_topk_ratio
        self.topk_beta = float(topk_beta)
        self.image_score_mode = str(image_score_mode).lower()
        if self.image_score_mode not in (
            "legacy",
            "evidence_mil",
            "reviewer_mil",
            "relative_reviewer_mil",
            "pdar_image_head",
        ):
            raise ValueError(
                "image_score_mode must be 'legacy', 'evidence_mil', "
                "'reviewer_mil', 'relative_reviewer_mil', or "
                "'pdar_image_head', "
                f"got {image_score_mode!r}."
            )
        self.image_reviewer_raw_temperature = float(
            image_reviewer_raw_temperature
        )
        self.image_reviewer_topk_ratio = image_reviewer_topk_ratio
        if self.image_reviewer_raw_temperature <= 0.0:
            raise ValueError("image_reviewer_raw_temperature must be positive.")
        if not 0.0 < float(self.image_reviewer_topk_ratio) <= 1.0:
            raise ValueError("image_reviewer_topk_ratio must be in (0, 1].")
        self.image_reviewer_weight_init = tuple(
            float(value) for value in image_reviewer_weight_init
        )
        if len(self.image_reviewer_weight_init) != 4:
            raise ValueError(
                "image_reviewer_weight_init must contain wg, wr, wa, and wd."
            )
        if any(value <= 0.0 for value in self.image_reviewer_weight_init):
            raise ValueError("All image reviewer initial weights must be positive.")
        self.image_relative_reviewer_margin = float(
            image_relative_reviewer_margin
        )
        self.image_relative_reviewer_max_scale = float(
            image_relative_reviewer_max_scale
        )
        self.image_relative_reviewer_scale_init = float(
            image_relative_reviewer_scale_init
        )
        self.image_relative_reviewer_base_weight_init = tuple(
            float(value) for value in image_relative_reviewer_base_weight_init
        )
        self.image_relative_reviewer_detach_mamba = bool(
            image_relative_reviewer_detach_mamba
        )
        if self.image_relative_reviewer_margin < 0.0:
            raise ValueError("image_relative_reviewer_margin must be non-negative.")
        if self.image_relative_reviewer_max_scale <= 0.0:
            raise ValueError("image_relative_reviewer_max_scale must be positive.")
        if not (
            0.0
            < self.image_relative_reviewer_scale_init
            < self.image_relative_reviewer_max_scale
        ):
            raise ValueError(
                "image_relative_reviewer_scale_init must be in "
                "(0, image_relative_reviewer_max_scale)."
            )
        if len(self.image_relative_reviewer_base_weight_init) != 4:
            raise ValueError(
                "image_relative_reviewer_base_weight_init must contain weights "
                "for global, raw max, raw top1, and raw top5 evidence."
            )
        if any(
            value <= 0.0
            for value in self.image_relative_reviewer_base_weight_init
        ):
            raise ValueError(
                "All relative reviewer base weights must be positive."
            )
        self.pdar_image_pool_temperature = max(float(pdar_image_pool_temperature), 1e-6)
        self.pdar_image_attention_detach = bool(pdar_image_attention_detach)
        self.pdar_image_loss_weight = float(pdar_image_loss_weight)
        self.arcc_mamba_context_scale = float(arcc_mamba_context_scale)
        self.arcc_inject_mamba = bool(arcc_inject_mamba)
        self.arcc_mamba_fusion_mode = str(arcc_mamba_fusion_mode).lower()
        if self.arcc_mamba_fusion_mode not in ("add", "concat"):
            raise ValueError(
                "arcc_mamba_fusion_mode must be 'add' or 'concat', "
                f"got {arcc_mamba_fusion_mode!r}."
            )
        self.arcc_mamba_feature_source = str(arcc_mamba_feature_source).lower()
        if self.arcc_mamba_feature_source not in ("context_tokens", "pdar_delta"):
            raise ValueError(
                "arcc_mamba_feature_source must be 'context_tokens' or 'pdar_delta', "
                f"got {arcc_mamba_feature_source!r}."
            )
        self.arcc_mamba_support_guidance = bool(arcc_mamba_support_guidance)
        self.arcc_mamba_support_detach = bool(arcc_mamba_support_detach)
        self.arcc_variant = str(arcc_variant).lower()
        if self.arcc_variant not in ("deformable", "pgsam_iterative"):
            raise ValueError(
                "arcc_variant must be 'deformable' or 'pgsam_iterative', "
                f"got {arcc_variant!r}."
            )
        if self.mamba_feature_contrast_target_weight > 0 and not (
            self.use_supervised_masks
            and self.use_mamba_context
            and self.arcc_variant == "pgsam_iterative"
        ):
            raise ValueError(
                "Mamba feature contrast requires supervised masks, Mamba context, "
                "and arcc_variant='pgsam_iterative'."
            )
        self.arcc_mode = str(arcc_mode).lower()
        if self.arcc_mode not in ("bidirectional", "mamba_veto"):
            raise ValueError(
                "arcc_mode must be 'bidirectional' or 'mamba_veto', "
                f"got {arcc_mode!r}."
            )
        if self.arcc_mode == "mamba_veto" and not (self.use_arcc and self.use_mamba_context):
            raise ValueError("arcc_mode='mamba_veto' requires use_arcc=True and use_mamba_context=True.")
        if self.arcc_inject_mamba and not (self.use_arcc and self.use_mamba_context):
            raise ValueError("arcc_inject_mamba=True requires use_arcc=True and use_mamba_context=True.")
        if self.arcc_mamba_support_guidance and not (
            self.use_arcc and self.use_mamba_context and self.arcc_mode == "mamba_veto"
        ):
            raise ValueError(
                "arcc_mamba_support_guidance=True requires use_arcc=True, "
                "use_mamba_context=True, and arcc_mode='mamba_veto'."
            )
        if self.arcc_variant == "pgsam_iterative" and not (
            self.use_arcc and self.use_mamba_context and self.arcc_mode == "mamba_veto"
        ):
            raise ValueError(
                "arcc_variant='pgsam_iterative' requires use_arcc=True, "
                "use_mamba_context=True, and arcc_mode='mamba_veto'."
            )
        if self.arcc_variant == "pgsam_iterative" and (
            self.arcc_inject_mamba or self.arcc_mamba_support_guidance
        ):
            raise ValueError(
                "PGSAM iterative ARCC requires legacy arcc_inject_mamba and "
                "arcc_mamba_support_guidance to remain False."
            )
        if self.image_score_mode in (
            "evidence_mil",
            "reviewer_mil",
            "relative_reviewer_mil",
        ) and (
            self.arcc_mode != "mamba_veto"
        ):
            raise ValueError(
                f"image_score_mode={self.image_score_mode!r} requires "
                "arcc_mode='mamba_veto'."
            )
        if self.image_score_mode == "pdar_image_head" and not self.use_mamba_context:
            raise ValueError("image_score_mode='pdar_image_head' requires use_mamba_context=True.")
        self.mamba_veto_source = str(mamba_veto_source).lower()
        if self.mamba_veto_source not in ("semantic", "prior"):
            raise ValueError(
                "mamba_veto_source must be 'semantic' or 'prior', "
                f"got {mamba_veto_source!r}."
            )
        if self.mamba_veto_source == "prior" and not self.use_mamba_context:
            raise ValueError("mamba_veto_source='prior' requires use_mamba_context=True.")
        self.mamba_veto_temperature = max(float(mamba_veto_temperature), 1e-6)
        self.mamba_veto_threshold = float(mamba_veto_threshold)
        self.mamba_veto_detach = bool(mamba_veto_detach)
        self.mamba_veto_enabled = bool(mamba_veto_enabled)
        alpha_init = float(mamba_veto_alpha_init)
        if not 0.0 < alpha_init < 1.0:
            raise ValueError(f"mamba_veto_alpha_init must be within (0, 1), got {alpha_init}.")
        if self.arcc_mode == "mamba_veto":
            alpha_logit = math.log(alpha_init / (1.0 - alpha_init))
            self.mamba_veto_alpha_logit = nn.Parameter(torch.tensor(alpha_logit))
            self.stage = f"{self.stage}_mambaveto"
        else:
            self.register_parameter("mamba_veto_alpha_logit", None)
        self.arcc_lambda_override = None
        if arcc_lambda_override is not None:
            self.arcc_lambda_override = float(arcc_lambda_override)
            if not 0.0 <= self.arcc_lambda_override <= 2.0:
                raise ValueError(
                    "arcc_lambda_override must be within [0, 2], "
                    f"got {self.arcc_lambda_override}."
                )
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
        self.arcc_mamba_projection = None
        self.arcc_mamba_fusion = None
        if self.arcc_inject_mamba:
            injection_init = float(arcc_mamba_injection_init)
            if not 0.0 < injection_init < 1.0:
                raise ValueError(
                    "arcc_mamba_injection_init must be within (0, 1), "
                    f"got {injection_init}."
                )
            if self.arcc_mamba_fusion_mode == "add":
                self.arcc_mamba_projection = nn.Conv2d(
                    embed_dim,
                    embed_dim,
                    kernel_size=1,
                    bias=False,
                )
                nn.init.eye_(self.arcc_mamba_projection.weight[:, :, 0, 0])
            else:
                # Concatenation keeps the two sources distinguishable until a
                # learned 1x1 projection decides how to mix their channels.
                # [I, I] plus the bounded gamma below starts from the familiar
                # CNN + gamma * PDAR correction without locking training to it.
                self.arcc_mamba_fusion = nn.Conv2d(
                    2 * embed_dim,
                    embed_dim,
                    kernel_size=1,
                    bias=False,
                )
                with torch.no_grad():
                    self.arcc_mamba_fusion.weight.zero_()
                    fusion_weight = self.arcc_mamba_fusion.weight[:, :, 0, 0]
                    nn.init.eye_(fusion_weight[:, :embed_dim])
                    nn.init.eye_(fusion_weight[:, embed_dim:])
            injection_logit = math.log(injection_init / (1.0 - injection_init))
            self.arcc_mamba_injection_logit = nn.Parameter(torch.tensor(injection_logit))
            self.stage = f"{self.stage}_jointarcc"
        else:
            self.register_parameter("arcc_mamba_injection_logit", None)
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
            arcc_kwargs = dict(arcc_kwargs or {})
            if self.arcc_variant == "pgsam_iterative":
                from model.mambaad import PGSAMIterativeARCC

                self.arcc = PGSAMIterativeARCC(
                    embed_dim,
                    cross_dim=int(arcc_kwargs.get("cross_dim", 128)),
                    num_heads=int(arcc_kwargs.get("num_heads", 4)),
                    num_refine_steps=int(arcc_kwargs.get("num_refine_steps", 2)),
                    gate_hidden_dim=int(arcc_kwargs.get("gate_hidden_dim", 64)),
                    gamma_init=float(arcc_kwargs.get("gamma_init", 0.05)),
                    rho_init=float(arcc_kwargs.get("rho_init", 0.1)),
                    rho_max=float(arcc_kwargs.get("rho_max", 0.5)),
                    eps=float(arcc_kwargs.get("eps", 1e-6)),
                    arcc_context_mode=str(
                        arcc_kwargs.get("arcc_context_mode", "real")
                    ),
                    normalize_cross_features=bool(
                        arcc_kwargs.get("normalize_cross_features", False)
                    ),
                    cross_norm_eps=float(
                        arcc_kwargs.get("cross_norm_eps", 1e-6)
                    ),
                    use_context_gate=bool(
                        arcc_kwargs.get("use_context_gate", True)
                    ),
                    use_dynamic_gate=bool(
                        arcc_kwargs.get("use_dynamic_gate", True)
                    ),
                )
                self.stage = f"{self.stage}_pgsamiterative"
            else:
                from model.mambaad import ARCCCalibration

                self.arcc_lambda = nn.Parameter(
                    torch.tensor(float(arcc_kwargs.get("lambda_init", 0.1)))
                )
                self.arcc = ARCCCalibration(
                    embed_dim,
                    use_response=bool(arcc_kwargs.get("use_response", True)),
                    use_mamba_support=self.arcc_mamba_support_guidance,
                    use_foreground=bool(arcc_kwargs.get("use_foreground", False)),
                    use_edge=bool(arcc_kwargs.get("use_edge", False)),
                    kernel_size=int(arcc_kwargs.get("kernel_size", 3)),
                    hidden_dim=arcc_kwargs.get("hidden_dim", None),
                    lambda_init=float(arcc_kwargs.get("lambda_init", 0.1)),
                )

        self.image_fusion_head = None
        self.image_reviewer_weight_logits = None
        self.image_reviewer_bias = None
        self.relative_reviewer_base_weight_logits = None
        self.relative_reviewer_bias = None
        self.relative_reviewer_agree_logit = None
        self.relative_reviewer_reject_logit = None
        self.pdar_image_head = None
        if self.image_score_mode == "evidence_mil":
            # Evidence order: global, raw max/top1/top5, Mamba max/top1/top5.
            self.image_fusion_head = nn.Linear(7, 1)
            with torch.no_grad():
                self.image_fusion_head.weight.zero_()
                self.image_fusion_head.weight[0, 0] = 1.0
                self.image_fusion_head.weight[0, 3] = 0.5
                self.image_fusion_head.weight[0, 6] = 0.5
                self.image_fusion_head.bias.zero_()
            self.stage = f"{self.stage}_imagemil"
        elif self.image_score_mode == "reviewer_mil":
            # Positive magnitudes are parameterized through softplus. The
            # disagreement magnitude is subtracted explicitly in forward(),
            # so Mamba disagreement can never accidentally become a bonus.
            initial_logits = [
                math.log(math.expm1(value))
                for value in self.image_reviewer_weight_init
            ]
            self.image_reviewer_weight_logits = nn.Parameter(
                torch.tensor(initial_logits, dtype=torch.float32)
            )
            self.image_reviewer_bias = nn.Parameter(torch.zeros(()))
            self.stage = f"{self.stage}_reviewermil"
        elif self.image_score_mode == "relative_reviewer_mil":
            # The base score contains no stand-alone Mamba evidence. Mamba is
            # only allowed to make a small, bounded correction after comparing
            # its support inside the raw candidate region against background.
            base_logits = [
                math.log(math.expm1(value))
                for value in self.image_relative_reviewer_base_weight_init
            ]
            self.relative_reviewer_base_weight_logits = nn.Parameter(
                torch.tensor(base_logits, dtype=torch.float32)
            )
            self.relative_reviewer_bias = nn.Parameter(torch.zeros(()))
            scale_ratio = (
                self.image_relative_reviewer_scale_init
                / self.image_relative_reviewer_max_scale
            )
            scale_logit = math.log(scale_ratio / (1.0 - scale_ratio))
            self.relative_reviewer_agree_logit = nn.Parameter(
                torch.tensor(scale_logit, dtype=torch.float32)
            )
            self.relative_reviewer_reject_logit = nn.Parameter(
                torch.tensor(scale_logit, dtype=torch.float32)
            )
            self.stage = f"{self.stage}_relreviewermil"
        elif self.image_score_mode == "pdar_image_head":
            hidden_dim = max(1, embed_dim // 4)
            self.pdar_image_head = nn.Sequential(
                nn.LayerNorm(2 * embed_dim),
                nn.Linear(2 * embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(pdar_image_dropout)),
                nn.Linear(hidden_dim, 1),
            )
            # The combined image score starts exactly from the V7 legacy path.
            # The auxiliary BCE can train the zero-initialized head safely.
            nn.init.zeros_(self.pdar_image_head[-1].weight)
            nn.init.zeros_(self.pdar_image_head[-1].bias)
            scale_init = float(pdar_image_scale_init)
            if not 0.0 < scale_init < 1.0:
                raise ValueError(
                    "pdar_image_scale_init must be within (0, 1), "
                    f"got {pdar_image_scale_init}."
                )
            scale_logit = math.log(scale_init / (1.0 - scale_init))
            self.pdar_image_scale_logit = nn.Parameter(torch.tensor(scale_logit))
            self.stage = f"{self.stage}_pdarimage"
        else:
            self.register_parameter("pdar_image_scale_logit", None)

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
        if self.arcc_mamba_projection is not None:
            self.arcc_mamba_projection.train(mode)
        if self.arcc_mamba_fusion is not None:
            self.arcc_mamba_fusion.train(mode)
        if self.image_fusion_head is not None:
            self.image_fusion_head.train(mode)
        if self.pdar_image_head is not None:
            self.pdar_image_head.train(mode)
        return self

    @staticmethod
    def _map_evidence(score_map):
        return (
            mean_topk_score(score_map, None),
            mean_topk_score(score_map, 0.01),
            mean_topk_score(score_map, 0.05),
        )

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

    def _apply_arcc(
        self,
        tokens,
        raw_patch_map,
        raw_anomaly_map,
        image_shape,
        protos,
        mamba_source_tokens=None,
    ):
        if self.arcc is None:
            return raw_anomaly_map, None, tokens
        context_tokens = tokens
        mamba_debug = {}
        mamba_tokens = None
        cnn_feature_map = self._tokens_to_feature_map(tokens)
        feature_map = cnn_feature_map
        injection_gamma = None
        mamba_semantic_patch = None
        mamba_semantic_logits = None
        mamba_verifier_logits = None
        mamba_support_guidance = None
        grid_h, grid_w = self._grid_size()
        if self.mamba_context is not None:
            mamba_source_tokens = tokens if mamba_source_tokens is None else mamba_source_tokens
            mamba_tokens, mamba_debug = self.mamba_context(
                mamba_source_tokens,
                self._grid_size(),
            )
            full_context_tokens = mamba_debug.pop(
                "mamba_full_context_tokens",
                mamba_tokens,
            )
            if self.arcc_mamba_feature_source == "pdar_delta":
                normalized_context = F.layer_norm(
                    full_context_tokens,
                    (full_context_tokens.shape[-1],),
                )
                normalized_raw = F.layer_norm(
                    mamba_source_tokens,
                    (mamba_source_tokens.shape[-1],),
                )
                mamba_injection_tokens = normalized_context - normalized_raw
            else:
                mamba_injection_tokens = mamba_tokens
            mamba_feature_map = self._tokens_to_feature_map(mamba_injection_tokens)
            global_context_feature_map = self._tokens_to_feature_map(full_context_tokens)
            if self.arcc_inject_mamba:
                injection_gamma = torch.sigmoid(self.arcc_mamba_injection_logit)
                if self.arcc_mamba_fusion_mode == "concat":
                    feature_map = self.arcc_mamba_fusion(
                        torch.cat(
                            (cnn_feature_map, injection_gamma * mamba_feature_map),
                            dim=1,
                        )
                    )
                else:
                    projected_mamba = self.arcc_mamba_projection(mamba_feature_map)
                    feature_map = cnn_feature_map + injection_gamma * projected_mamba
                context_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
            elif self.arcc_mode == "bidirectional":
                feature_map = feature_map + self.arcc_mamba_context_scale * mamba_feature_map
                context_tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
            else:
                # The verifier stays independent of the external CNN adapter
                # and is not injected back into ARCC's feature stream.
                context_tokens = mamba_tokens

            if self.arcc_mode == "mamba_veto":
                mamba_semantic_patch, _ = abnormal_minus_normal(mamba_tokens, protos)
                mamba_semantic_logits = mamba_semantic_patch.reshape(
                    mamba_semantic_patch.shape[0], grid_h, grid_w
                )
                if self.mamba_veto_source == "prior":
                    mamba_verifier_logits = mamba_debug["mamba_context_logits"]
                else:
                    mamba_verifier_logits = mamba_semantic_logits
                if getattr(self, "arcc_mamba_support_guidance", False):
                    mamba_support_guidance = torch.sigmoid(
                        (mamba_verifier_logits - self.mamba_veto_threshold)
                        / self.mamba_veto_temperature
                    )
                    if getattr(self, "arcc_mamba_support_detach", True):
                        mamba_support_guidance = mamba_support_guidance.detach()

        if self.arcc_variant == "pgsam_iterative":
            if mamba_tokens is None or mamba_verifier_logits is None:
                raise RuntimeError(
                    "PGSAM iterative ARCC requires PDAR context and the unchanged "
                    "Evidence-MIL verifier path."
                )
            final_patch_logits, iterative_debug = self.arcc(
                cnn_feature_map,
                global_context_feature_map,
                raw_patch_map,
            )
            final_map = F.interpolate(
                final_patch_logits,
                size=image_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            # The semantic/prior maps below are retained only for V18's
            # independent Mamba losses, diagnostics, and Evidence-MIL image
            # score. They do not enter the new pixel-level ARCC.
            mamba_semantic_map = upsample_patch_map(
                mamba_semantic_patch,
                grid_size=(grid_h, grid_w),
                image_shape=image_shape,
            )
            if self.mamba_veto_source == "prior":
                mamba_verifier_map = upsample_patch_map(
                    mamba_verifier_logits.flatten(1),
                    grid_size=(grid_h, grid_w),
                    image_shape=image_shape,
                )
            else:
                mamba_verifier_map = mamba_semantic_map
            mamba_support = torch.sigmoid(
                (mamba_verifier_map - self.mamba_veto_threshold)
                / self.mamba_veto_temperature
            )
            debug = {
                **iterative_debug,
                # Compatibility-only: the context gate is not a deformable
                # modulation mask and is therefore exposed under its own name.
                "arcc_mod_mask": None,
                "A_final_patch": final_patch_logits.squeeze(1),
                "arcc_suppressed_map": final_map,
                "mamba_context_tokens": mamba_tokens,
                "mamba_semantic_logits": mamba_semantic_logits,
                "mamba_semantic_map": mamba_semantic_map,
                "mamba_verifier_logits": mamba_verifier_logits,
                "mamba_verifier_map": mamba_verifier_map,
                "mamba_support_map": mamba_support,
                "mamba_veto_map": torch.zeros_like(final_map),
                "mamba_veto_alpha": final_map.new_zeros(()),
                # Private training-only tensor consumed before outputs are
                # exported. This is the exact high-dimensional PDAR context
                # used as K/V by the QKV fusion path.
                "_mamba_contrast_tokens": full_context_tokens,
                **mamba_debug,
            }
            if self.pdar_image_head is not None:
                debug["_pdar_image_tokens"] = full_context_tokens
            return final_map, debug, mamba_tokens

        arcc_guidance_kwargs = {}
        if mamba_support_guidance is not None:
            arcc_guidance_kwargs["mamba_support"] = mamba_support_guidance
        g_cal, mod_mask = self.arcc(
            feature_map,
            raw_patch_map,
            foreground=None,
            edge=None,
            image_shape=image_shape,
            **arcc_guidance_kwargs,
        )
        learned_arcc_lambda = torch.clamp(self.arcc_lambda, min=0.0, max=2.0)
        if self.arcc_lambda_override is None:
            arcc_lambda = learned_arcc_lambda
        else:
            # Diagnostic ablation only: 0 makes A_final exactly equal A_raw.
            arcc_lambda = learned_arcc_lambda.new_tensor(self.arcc_lambda_override)
        bidirectional_map = raw_anomaly_map + arcc_lambda * raw_anomaly_map * torch.tanh(g_cal)
        debug = {
            "G_cal": g_cal,
            "arcc_mod_mask": mod_mask,
            "arcc_lambda": arcc_lambda,
            "arcc_lambda_learned": learned_arcc_lambda,
            "arcc_bidirectional_map": bidirectional_map,
            "mamba_context_tokens": context_tokens,
            **mamba_debug,
        }
        if self.pdar_image_head is not None:
            # Consumed and removed by forward(); never exported as a full
            # per-image tensor by the evaluator.
            debug["_pdar_image_tokens"] = full_context_tokens
        if injection_gamma is not None:
            debug["arcc_mamba_injection_gamma"] = injection_gamma
        if mamba_support_guidance is not None:
            debug["arcc_mamba_support_guidance"] = mamba_support_guidance
        if self.arcc_mode == "bidirectional":
            return bidirectional_map, debug, context_tokens

        mamba_semantic_map = upsample_patch_map(
            mamba_semantic_patch,
            grid_size=(grid_h, grid_w),
            image_shape=image_shape,
        )
        if self.mamba_veto_source == "prior":
            mamba_verifier_patch = mamba_verifier_logits.flatten(1)
            mamba_verifier_map = upsample_patch_map(
                mamba_verifier_patch,
                grid_size=(grid_h, grid_w),
                image_shape=image_shape,
            )
        else:
            mamba_verifier_logits = mamba_semantic_logits
            mamba_verifier_map = mamba_semantic_map
        mamba_support = torch.sigmoid(
            (mamba_verifier_map - self.mamba_veto_threshold)
            / self.mamba_veto_temperature
        )
        if getattr(self, "mamba_veto_enabled", True):
            veto_support = mamba_support.detach() if self.mamba_veto_detach else mamba_support
            veto_alpha = torch.sigmoid(self.mamba_veto_alpha_logit)
            final_map, suppressed_map, veto_map = apply_mamba_probability_veto(
                raw_anomaly_map,
                bidirectional_map,
                veto_support,
                veto_alpha,
            )
        else:
            # ARCC-only ablation: retain exactly the same Mamba evidence and
            # support guidance used by evidence-MIL, but do not apply the
            # no-amplification clamp or the post-ARCC probability veto.
            final_map = bidirectional_map
            suppressed_map = bidirectional_map
            veto_map = torch.zeros_like(bidirectional_map)
            veto_alpha = bidirectional_map.new_zeros(())
        debug.update(
            {
                "arcc_suppressed_map": suppressed_map,
                "mamba_semantic_logits": mamba_semantic_logits,
                "mamba_semantic_map": mamba_semantic_map,
                "mamba_verifier_logits": mamba_verifier_logits,
                "mamba_verifier_map": mamba_verifier_map,
                "mamba_support_map": mamba_support,
                "mamba_veto_map": veto_map,
                "mamba_veto_alpha": veto_alpha,
            }
        )
        if self.arcc_inject_mamba and not self.training:
            # Evaluation-only counterfactual: reuse the exact ARCC and veto
            # weights, but remove Mamba feature injection from ARCC's input.
            # This isolates the contribution of the joint feature stream.
            cnn_only_g_cal, _ = self.arcc(
                cnn_feature_map,
                raw_patch_map,
                foreground=None,
                edge=None,
                image_shape=image_shape,
                **arcc_guidance_kwargs,
            )
            cnn_only_bidirectional = (
                raw_anomaly_map
                + arcc_lambda * raw_anomaly_map * torch.tanh(cnn_only_g_cal)
            )
            cnn_only_final, _, _ = apply_mamba_probability_veto(
                raw_anomaly_map,
                cnn_only_bidirectional,
                veto_support,
                veto_alpha,
            )
            debug["arcc_cnn_only_final_map"] = cnn_only_final
        return final_map, debug, context_tokens

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
        targets = resize_mamba_patch_targets(
            targets,
            output_size=context_logits.shape[-2:],
            mode=self.mamba_context_mask_pool,
        )
        logits = context_logits.unsqueeze(1) / self.supervised_score_temperature
        loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
        loss_dice = self._dice_loss(logits, targets)
        loss_outside = self._outside_topk_loss_with_ratio(
            context_logits,
            targets,
            self.mamba_context_outside_topk_ratio,
        )
        return loss_bce, loss_dice, loss_outside

    def _mamba_context_separation_loss(self, context_logits, masks):
        """Require positive-mask logits to exceed the hardest outside logits."""
        targets = masks.to(device=context_logits.device, dtype=context_logits.dtype)
        targets = resize_mamba_patch_targets(
            targets,
            output_size=context_logits.shape[-2:],
            mode=self.mamba_context_mask_pool,
        ).squeeze(1)
        losses = []
        for logits_per_image, target_per_image in zip(context_logits, targets):
            inside = target_per_image > 0.5
            outside = ~inside
            if not inside.any() or not outside.any():
                continue
            inside_mean = logits_per_image[inside].mean()
            outside_values = logits_per_image[outside]
            if self.mamba_context_outside_topk_ratio is None:
                outside_topk = outside_values.max()
            else:
                topk = max(
                    1,
                    int(outside_values.numel() * float(self.mamba_context_outside_topk_ratio)),
                )
                outside_topk = outside_values.topk(topk).values.mean()
            losses.append(
                F.relu(
                    self.mamba_context_separation_margin
                    + outside_topk
                    - inside_mean
                )
            )
        if not losses:
            return context_logits.new_zeros(())
        return torch.stack(losses).mean()

    def _mamba_feature_contrast_weight(self, current_epoch):
        """Return the deterministic epoch-wise contrast weight.

        The trainer passes a one-based epoch, so target=0.2 and warmup=2
        produce 0.1, 0.2, 0.2, ... for epochs 1, 2, 3, ... . This is not a
        parameter and therefore cannot be reduced by gradient descent.
        """
        target = self.mamba_feature_contrast_target_weight
        if target <= 0.0:
            return 0.0
        if self.mamba_feature_contrast_warmup_epochs <= 0:
            return target
        # A direct model call that omits the epoch is treated as epoch 1,
        # never as a zero-weight learnable shortcut.
        epoch = max(1.0, float(current_epoch or 1.0))
        progress = min(
            1.0,
            epoch / float(self.mamba_feature_contrast_warmup_epochs),
        )
        return target * progress

    def _mamba_feature_contrast_loss(
        self,
        context_tokens,
        context_logits,
        masks,
    ):
        """Contrast abnormal patches with hard normal patches in PDAR space.

        Contrast is computed independently inside every abnormal image. This
        avoids collapsing normal textures from unrelated object categories
        into one global prototype. The hardest mask-out patches are selected
        by the detached verifier logits so the loss focuses on false-positive
        normal structures without letting the selector itself carry gradient.
        """
        if context_tokens.ndim != 3:
            raise ValueError("Mamba contrast tokens must have shape [B, N, D].")
        if context_logits.ndim != 3:
            raise ValueError("Mamba verifier logits must have shape [B, H, W].")
        if context_logits.shape[0] != context_tokens.shape[0]:
            raise ValueError("Mamba contrast tokens/logits batch sizes must match.")
        if context_logits.shape[-2] * context_logits.shape[-1] != context_tokens.shape[1]:
            raise ValueError("Mamba contrast token count must match the verifier grid.")

        targets = masks.to(device=context_tokens.device, dtype=context_tokens.dtype)
        targets = resize_mamba_patch_targets(
            targets,
            output_size=context_logits.shape[-2:],
            mode=self.mamba_context_mask_pool,
        ).squeeze(1)
        target_flat = targets.flatten(1) > 0.5
        verifier_flat = context_logits.detach().flatten(1)

        # Float32 cosine logits remain stable under AMP; gradients still flow
        # back into the original PDAR/HSS context through the cast operation.
        features = F.normalize(context_tokens.float(), dim=-1)
        losses = []
        contrast_gaps = []
        prototype_cosines = []
        hard_negative_ratio = self.mamba_feature_contrast_hard_negative_ratio
        temperature = self.mamba_feature_contrast_temperature

        for feature_i, verifier_i, target_i in zip(
            features,
            verifier_flat,
            target_flat,
        ):
            inside = target_i
            outside = ~inside
            if not inside.any() or not outside.any():
                continue

            positive_features = feature_i[inside]
            outside_features = feature_i[outside]
            outside_scores = verifier_i[outside]
            hard_negative_count = max(
                1,
                int(outside_features.shape[0] * hard_negative_ratio),
            )
            # At least match the positive count when possible, while retaining
            # the configured top-ratio floor for tiny anomaly masks.
            hard_negative_count = max(
                hard_negative_count,
                min(positive_features.shape[0], outside_features.shape[0]),
            )
            hard_negative_count = min(
                hard_negative_count,
                outside_features.shape[0],
            )
            hard_indices = outside_scores.topk(hard_negative_count).indices
            negative_features = outside_features[hard_indices]

            abnormal_prototype = F.normalize(
                positive_features.mean(dim=0),
                dim=0,
            )
            normal_prototype = F.normalize(
                negative_features.mean(dim=0),
                dim=0,
            )

            positive_logits = torch.stack(
                (
                    positive_features @ normal_prototype,
                    positive_features @ abnormal_prototype,
                ),
                dim=1,
            ) / temperature
            negative_logits = torch.stack(
                (
                    negative_features @ normal_prototype,
                    negative_features @ abnormal_prototype,
                ),
                dim=1,
            ) / temperature
            positive_labels = torch.ones(
                positive_logits.shape[0],
                device=positive_logits.device,
                dtype=torch.long,
            )
            negative_labels = torch.zeros(
                negative_logits.shape[0],
                device=negative_logits.device,
                dtype=torch.long,
            )
            # Equal class weighting prevents the much larger normal region
            # from overwhelming small anomalous regions.
            loss_i = 0.5 * (
                F.cross_entropy(positive_logits, positive_labels)
                + F.cross_entropy(negative_logits, negative_labels)
            )
            losses.append(loss_i)
            contrast_gaps.append(
                (positive_logits[:, 1] - positive_logits[:, 0]).mean()
                * temperature
            )
            prototype_cosines.append(
                torch.sum(abnormal_prototype * normal_prototype)
            )

        if not losses:
            zero = context_tokens.new_zeros((), dtype=torch.float32)
            return zero, zero, zero
        return (
            torch.stack(losses).mean(),
            torch.stack(contrast_gaps).mean(),
            torch.stack(prototype_cosines).mean(),
        )

    def _adapter_losses(
        self,
        raw_patch,
        refined_patch,
        patch_score,
        image_score,
        pdar_image_logit=None,
        anomaly_map=None,
        raw_anomaly_map=None,
        arcc_debug=None,
        mamba_contrast_tokens=None,
        masks=None,
        labels=None,
        current_epoch=None,
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
        loss_pdar_image = image_score.new_zeros(())
        if self.use_supervised_masks and labels is not None:
            image_target = labels.to(device=image_score.device, dtype=image_score.dtype).view_as(image_score)
            loss_image_supervised = F.binary_cross_entropy_with_logits(image_score, image_target)
            if pdar_image_logit is not None:
                loss_pdar_image = F.binary_cross_entropy_with_logits(
                    pdar_image_logit,
                    image_target,
                )
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
        loss_mamba_context_separation = image_score.new_zeros(())
        loss_mamba_feature_contrast = image_score.new_zeros(())
        dbg_mamba_feature_contrast_gap = image_score.new_zeros(())
        dbg_mamba_feature_prototype_cosine = image_score.new_zeros(())
        mamba_feature_contrast_weight = image_score.new_tensor(
            self._mamba_feature_contrast_weight(current_epoch)
        )
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
            and "mamba_verifier_logits" in arcc_debug
            and (
                self.mamba_context_bce_weight > 0
                or self.mamba_context_dice_weight > 0
                or self.mamba_context_outside_topk_weight > 0
                or self.mamba_context_separation_weight > 0
            )
        ):
            (
                loss_mamba_context_bce,
                loss_mamba_context_dice,
                loss_mamba_context_outside_topk,
            ) = self._mamba_context_losses(arcc_debug["mamba_verifier_logits"], masks)
            if self.mamba_context_separation_weight > 0:
                loss_mamba_context_separation = self._mamba_context_separation_loss(
                    arcc_debug["mamba_verifier_logits"],
                    masks,
                )
        if (
            self.use_supervised_masks
            and masks is not None
            and arcc_debug is not None
            and "mamba_verifier_logits" in arcc_debug
            and mamba_contrast_tokens is not None
            and self.mamba_feature_contrast_target_weight > 0
        ):
            (
                loss_mamba_feature_contrast,
                dbg_mamba_feature_contrast_gap,
                dbg_mamba_feature_prototype_cosine,
            ) = self._mamba_feature_contrast_loss(
                mamba_contrast_tokens,
                arcc_debug["mamba_verifier_logits"],
                masks,
            )
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
            + self.pdar_image_loss_weight * loss_pdar_image
            + self.mamba_context_bce_weight * loss_mamba_context_bce
            + self.mamba_context_dice_weight * loss_mamba_context_dice
            + self.mamba_context_outside_topk_weight * loss_mamba_context_outside_topk
            + self.mamba_context_separation_weight * loss_mamba_context_separation
            + mamba_feature_contrast_weight * loss_mamba_feature_contrast
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
            "loss_pdar_image": loss_pdar_image,
            "loss_mamba_context_bce": loss_mamba_context_bce,
            "loss_mamba_context_dice": loss_mamba_context_dice,
            "loss_mamba_context_outside_topk": loss_mamba_context_outside_topk,
            "loss_mamba_context_separation": loss_mamba_context_separation,
            "loss_mamba_feature_contrast": loss_mamba_feature_contrast,
            "dbg_mamba_feature_contrast_weight": mamba_feature_contrast_weight,
            "dbg_mamba_feature_contrast_gap": dbg_mamba_feature_contrast_gap,
            "dbg_mamba_feature_prototype_cosine": dbg_mamba_feature_prototype_cosine,
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

            # Pure diagnostics: split normal and anomalous-mask behavior without
            # changing any loss term or gradient used for optimization.
            if masks is not None:
                target = masks.to(device=anomaly_map.device, dtype=anomaly_map.dtype)
                if target.ndim == 4:
                    target = target.squeeze(1)
                if target.shape[-2:] != anomaly_map.shape[-2:]:
                    target = F.interpolate(
                        target.unsqueeze(1),
                        size=anomaly_map.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                target = target.clamp(0.0, 1.0)
                # Match the evaluator's actual binary-mask rule. Fractional
                # interpolation residue below 0.5 must not count as a valid mask.
                positive_mask = (target.flatten(1) > 0.5).any(dim=1)
                if labels is not None:
                    label_values = labels.to(device=anomaly_map.device).view(-1)
                    normal_images = label_values == 0
                    abnormal_images = label_values != 0
                else:
                    normal_images = ~positive_mask
                    abnormal_images = positive_mask

                final_logits = anomaly_map.detach() / self.supervised_score_temperature
                raw_scores = raw_anomaly_map.detach()
                final_scores = anomaly_map.detach()
                bce_per_image = F.binary_cross_entropy_with_logits(
                    final_logits,
                    target,
                    reduction="none",
                ).flatten(1).mean(dim=1)
                probs = torch.sigmoid(final_logits)
                probs_flat = probs.flatten(1)
                target_flat = target.flatten(1)
                intersection = (probs_flat * target_flat).sum(dim=1)
                dice_loss_per_image = 1.0 - (
                    (2.0 * intersection + 1.0)
                    / (probs_flat.sum(dim=1) + target_flat.sum(dim=1) + 1.0)
                )

                def selected_mean(values, selection):
                    return values[selection].mean() if selection.any() else values.new_zeros(())

                debug["dbg_mask_bce_normal"] = selected_mean(bce_per_image, normal_images)
                debug["dbg_mask_bce_abnormal"] = selected_mean(bce_per_image, abnormal_images)
                debug["dbg_mask_dice_positive"] = selected_mean(
                    dice_loss_per_image,
                    positive_mask,
                )
                normal_prob_per_image = probs.flatten(1).mean(dim=1)
                normal_fg_per_image = (probs.flatten(1) > 0.5).float().mean(dim=1)
                debug["dbg_normal_prob_mean"] = selected_mean(
                    normal_prob_per_image,
                    normal_images,
                )
                debug["dbg_normal_fg_ratio"] = selected_mean(
                    normal_fg_per_image,
                    normal_images,
                )
                raw_max = raw_scores.flatten(1).amax(dim=1)
                final_max = final_scores.flatten(1).amax(dim=1)
                debug["dbg_arcc_normal_max_gain"] = selected_mean(
                    final_max - raw_max,
                    normal_images,
                )

                binary_target = (target > 0.5).to(dtype=target.dtype)
                outside_target = 1.0 - binary_target
                inside_count = binary_target.flatten(1).sum(dim=1).clamp_min(1.0)
                outside_count = outside_target.flatten(1).sum(dim=1).clamp_min(1.0)

                def mask_gap(scores):
                    inside_mean = (scores * binary_target).flatten(1).sum(dim=1) / inside_count
                    outside_mean = (scores * outside_target).flatten(1).sum(dim=1) / outside_count
                    return inside_mean - outside_mean

                debug["dbg_raw_mask_gap"] = selected_mean(mask_gap(raw_scores), positive_mask)
                debug["dbg_final_mask_gap"] = selected_mean(mask_gap(final_scores), positive_mask)
                if arcc_debug is not None and "arcc_cnn_only_final_map" in arcc_debug:
                    cnn_only_scores = arcc_debug["arcc_cnn_only_final_map"].detach()
                    joint_minus_cnn = final_scores - cnn_only_scores
                    cnn_only_max = cnn_only_scores.flatten(1).amax(dim=1)
                    joint_max = final_scores.flatten(1).amax(dim=1)
                    max_delta = joint_max - cnn_only_max
                    debug["dbg_joint_vs_cnn_max_normal"] = selected_mean(
                        max_delta,
                        normal_images,
                    )
                    debug["dbg_joint_vs_cnn_max_abnormal"] = selected_mean(
                        max_delta,
                        abnormal_images,
                    )
                    inside_delta = (
                        (joint_minus_cnn * binary_target).flatten(1).sum(dim=1)
                        / inside_count
                    )
                    outside_delta = (
                        (joint_minus_cnn * outside_target).flatten(1).sum(dim=1)
                        / outside_count
                    )
                    debug["dbg_joint_vs_cnn_mask_in"] = selected_mean(
                        inside_delta,
                        positive_mask,
                    )
                    debug["dbg_joint_vs_cnn_mask_out"] = selected_mean(
                        outside_delta,
                        positive_mask,
                    )
                debug["dbg_batch_normal_count"] = normal_images.float().sum()
                debug["dbg_batch_abnormal_count"] = abnormal_images.float().sum()
                debug["dbg_batch_positive_mask_count"] = positive_mask.float().sum()
            if arcc_debug is not None:
                if "arcc_lambda" in arcc_debug:
                    debug["dbg_arcc_lambda"] = arcc_debug["arcc_lambda"].detach()
                    debug["dbg_arcc_lambda_learned"] = arcc_debug[
                        "arcc_lambda_learned"
                    ].detach()
                else:
                    debug["dbg_arcc_lambda"] = raw_anomaly_map.new_zeros(())
                    debug["dbg_arcc_lambda_learned"] = raw_anomaly_map.new_zeros(())
                debug["dbg_g_cal_abs"] = arcc_debug["G_cal"].detach().abs().mean()
                if "cross_gamma" in arcc_debug:
                    debug["dbg_arcc_cross_gamma"] = arcc_debug[
                        "cross_gamma"
                    ].detach()
                    calibration_delta = arcc_debug["calibration_delta"].detach()
                    context_gate = arcc_debug.get("context_gate")
                    if context_gate is not None:
                        context_gate = context_gate.detach()
                        debug["dbg_arcc_context_gate_mean"] = context_gate.mean()
                        debug["dbg_arcc_context_gate_max"] = context_gate.amax()
                    debug["dbg_arcc_cross_feature_norm"] = arcc_debug[
                        "cross_feature_norm"
                    ].detach()
                    debug["dbg_arcc_local_context_difference"] = arcc_debug[
                        "local_context_difference"
                    ].detach()
                    if "dynamic_gate_norm" in arcc_debug:
                        debug["dbg_arcc_dynamic_gate_norm"] = arcc_debug[
                            "dynamic_gate_norm"
                        ].detach()
                    for step_idx in (1, 2):
                        debug[f"dbg_arcc_refine_step{step_idx}_abs"] = arcc_debug[
                            f"refine_step{step_idx}_abs"
                        ].detach()
                        debug[f"dbg_arcc_refine_step{step_idx}_signed_mean"] = arcc_debug[
                            f"refine_step{step_idx}_signed_mean"
                        ].detach()
                    debug["dbg_arcc_final_minus_raw_abs"] = arcc_debug[
                        "final_minus_raw_abs"
                    ].detach()
                    debug["dbg_arcc_final_minus_raw_signed_mean"] = arcc_debug[
                        "final_minus_raw_signed_mean"
                    ].detach()

                    if masks is not None:
                        patch_target = masks.to(
                            device=calibration_delta.device,
                            dtype=calibration_delta.dtype,
                        )
                        if patch_target.ndim == 3:
                            patch_target = patch_target.unsqueeze(1)
                        if patch_target.shape[-2:] != calibration_delta.shape[-2:]:
                            patch_target = F.adaptive_max_pool2d(
                                patch_target,
                                output_size=calibration_delta.shape[-2:],
                            )
                        patch_inside = patch_target > 0.5
                        patch_outside = ~patch_inside

                        def region_mean(values, selection):
                            return (
                                values[selection].mean()
                                if selection.any()
                                else values.new_zeros(())
                            )

                        debug["dbg_arcc_delta_inside"] = region_mean(
                            calibration_delta,
                            patch_inside,
                        )
                        debug["dbg_arcc_delta_outside"] = region_mean(
                            calibration_delta,
                            patch_outside,
                        )
                        if context_gate is not None:
                            debug["dbg_arcc_gate_inside"] = region_mean(
                                context_gate,
                                patch_inside,
                            )
                            debug["dbg_arcc_gate_outside"] = region_mean(
                                context_gate,
                                patch_outside,
                            )
                        raw_flat = raw_anomaly_map.detach().flatten(1)
                        final_flat = anomaly_map.detach().flatten(1)
                        normal_topk = max(1, int(raw_flat.shape[1] * 0.01))
                        raw_normal_topk = raw_flat.topk(normal_topk, dim=1).values.mean(dim=1)
                        final_normal_topk = final_flat.topk(normal_topk, dim=1).values.mean(dim=1)
                        debug["dbg_arcc_normal_topk_before"] = selected_mean(
                            raw_normal_topk,
                            normal_images,
                        )
                        debug["dbg_arcc_normal_topk_after"] = selected_mean(
                            final_normal_topk,
                            normal_images,
                        )
                if "arcc_mamba_injection_gamma" in arcc_debug:
                    debug["dbg_arcc_mamba_injection_gamma"] = arcc_debug[
                        "arcc_mamba_injection_gamma"
                    ].detach()
                if "arcc_mamba_support_guidance" in arcc_debug:
                    debug["dbg_arcc_mamba_support_guidance_mean"] = arcc_debug[
                        "arcc_mamba_support_guidance"
                    ].detach().mean()
                if "mamba_semantic_map" in arcc_debug:
                    semantic_map = arcc_debug["mamba_semantic_map"].detach()
                    verifier_map = arcc_debug["mamba_verifier_map"].detach()
                    support_map = arcc_debug["mamba_support_map"].detach()
                    veto_map = arcc_debug["mamba_veto_map"].detach()
                    debug["dbg_mamba_semantic_mean"] = semantic_map.mean()
                    debug["dbg_mamba_semantic_max"] = semantic_map.amax()
                    debug["dbg_mamba_verifier_mean"] = verifier_map.mean()
                    debug["dbg_mamba_verifier_max"] = verifier_map.amax()
                    debug["dbg_mamba_support_mean"] = support_map.mean()
                    debug["dbg_mamba_veto_mean"] = veto_map.mean()
                    debug["dbg_mamba_veto_alpha"] = arcc_debug[
                        "mamba_veto_alpha"
                    ].detach()
                    debug["dbg_mamba_veto_max_gain"] = (
                        anomaly_map.detach() - raw_anomaly_map.detach()
                    ).amax()
                    if masks is not None:
                        veto_target = masks.to(
                            device=support_map.device,
                            dtype=support_map.dtype,
                        )
                        if veto_target.ndim == 4:
                            veto_target = veto_target.squeeze(1)
                        if veto_target.shape[-2:] != support_map.shape[-2:]:
                            veto_target = F.interpolate(
                                veto_target.unsqueeze(1),
                                size=support_map.shape[-2:],
                                mode="nearest",
                            ).squeeze(1)
                        veto_positive = (veto_target.flatten(1) > 0.5).any(dim=1)
                        if labels is not None:
                            veto_normal = labels.to(device=support_map.device).view(-1) == 0
                        else:
                            veto_normal = ~veto_positive

                        def selected_region_mean(values, selection):
                            return values[selection].mean() if selection.any() else values.new_zeros(())

                        debug["dbg_mamba_support_normal"] = selected_region_mean(
                            support_map.flatten(1).mean(dim=1), veto_normal
                        )
                        debug["dbg_mamba_veto_normal"] = selected_region_mean(
                            veto_map.flatten(1).mean(dim=1), veto_normal
                        )
                        inside_mask = veto_target > 0.5
                        outside_mask = ~inside_mask
                        inside_count = inside_mask.flatten(1).sum(dim=1).clamp_min(1)
                        outside_count = outside_mask.flatten(1).sum(dim=1).clamp_min(1)
                        for name, values in (("support", support_map), ("veto", veto_map)):
                            inside_mean = (
                                (values * inside_mask).flatten(1).sum(dim=1) / inside_count
                            )
                            outside_mean = (
                                (values * outside_mask).flatten(1).sum(dim=1) / outside_count
                            )
                            debug[f"dbg_mamba_{name}_mask_in"] = selected_region_mean(
                                inside_mean, veto_positive
                            )
                            debug[f"dbg_mamba_{name}_mask_out"] = selected_region_mean(
                                outside_mean, veto_positive
                            )
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
                if "mamba_depth_stage_weight_means" in arcc_debug:
                    for stage_idx, stage_means in enumerate(
                        arcc_debug["mamba_depth_stage_weight_means"], start=1
                    ):
                        for source_idx, weight in enumerate(stage_means.detach().mean(dim=0)):
                            debug[f"dbg_mamba_s{stage_idx}_w_f{source_idx}"] = weight
            else:
                debug["dbg_arcc_lambda"] = raw_anomaly_map.new_zeros(())
                debug["dbg_g_cal_abs"] = raw_anomaly_map.new_zeros(())
            return debug

    def forward(
        self,
        imgs,
        cls_names=None,
        masks=None,
        labels=None,
        return_loss=True,
        current_epoch=None,
    ):
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
            protos=protos,
            mamba_source_tokens=raw_patch_feat,
        )
        pdar_image_tokens = None
        mamba_contrast_tokens = None
        if arcc_debug is not None:
            pdar_image_tokens = arcc_debug.pop("_pdar_image_tokens", None)
            mamba_contrast_tokens = arcc_debug.pop("_mamba_contrast_tokens", None)
        topk_score = mean_topk_score(anomaly_map, self.image_score_topk_ratio)
        topk_score_max, topk_score_top1, topk_score_top5 = self._map_evidence(anomaly_map)
        raw_score_max, raw_score_top1, raw_score_top5 = self._map_evidence(raw_anomaly_map)
        if arcc_debug is not None and "mamba_verifier_logits" in arcc_debug:
            mamba_score_max, mamba_score_top1, mamba_score_top5 = self._map_evidence(
                arcc_debug["mamba_verifier_logits"]
            )
        else:
            mamba_score_max = s_global.new_zeros(s_global.shape)
            mamba_score_top1 = s_global.new_zeros(s_global.shape)
            mamba_score_top5 = s_global.new_zeros(s_global.shape)
        image_evidence = torch.stack(
            (
                s_global,
                raw_score_max,
                raw_score_top1,
                raw_score_top5,
                mamba_score_max,
                mamba_score_top1,
                mamba_score_top5,
            ),
            dim=1,
        )
        reviewer_evidence = None
        relative_reviewer_evidence = None
        relative_reviewer_candidate_support = None
        relative_reviewer_background_support = None
        relative_reviewer_gap = None
        relative_reviewer_agree = None
        relative_reviewer_reject = None
        relative_reviewer_neutral = None
        relative_reviewer_base_score = None
        relative_reviewer_score_delta = None
        if self.image_score_mode == "reviewer_mil":
            if arcc_debug is None or "mamba_support_map" not in arcc_debug:
                raise RuntimeError(
                    "reviewer_mil requires a full-resolution Mamba support map."
                )

            # R is the CLIP/local-branch candidate probability. M is Mamba's
            # spatial support probability. Agreement and disagreement are
            # formed pixel by pixel before pooling, so support at a different
            # location cannot incorrectly approve a raw candidate.
            raw_candidate_map = torch.sigmoid(
                raw_anomaly_map / self.image_reviewer_raw_temperature
            )
            mamba_support_map = arcc_debug["mamba_support_map"]
            if mamba_support_map.shape[-2:] != raw_candidate_map.shape[-2:]:
                mamba_support_map = F.interpolate(
                    mamba_support_map.unsqueeze(1),
                    size=raw_candidate_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            mamba_support_map = mamba_support_map.clamp(0.0, 1.0)
            agree_map = raw_candidate_map * mamba_support_map
            disagree_map = raw_candidate_map * (1.0 - mamba_support_map)

            reviewer_raw = mean_topk_score(
                raw_candidate_map,
                self.image_reviewer_topk_ratio,
            )
            reviewer_mamba = mean_topk_score(
                mamba_support_map,
                self.image_reviewer_topk_ratio,
            )
            reviewer_agree = mean_topk_score(
                agree_map,
                self.image_reviewer_topk_ratio,
            )
            reviewer_disagree = mean_topk_score(
                disagree_map,
                self.image_reviewer_topk_ratio,
            )
            reviewer_evidence = torch.stack(
                (s_global, reviewer_raw, reviewer_agree, reviewer_disagree),
                dim=1,
            )
            raw_centered = raw_candidate_map.flatten(1)
            raw_centered = raw_centered - raw_centered.mean(dim=1, keepdim=True)
            mamba_centered = mamba_support_map.flatten(1)
            mamba_centered = mamba_centered - mamba_centered.mean(
                dim=1,
                keepdim=True,
            )
            reviewer_raw_mamba_corr = (
                (raw_centered * mamba_centered).sum(dim=1)
                / (
                    raw_centered.square().sum(dim=1).sqrt()
                    * mamba_centered.square().sum(dim=1).sqrt()
                ).clamp_min(1e-6)
            )
        elif self.image_score_mode == "relative_reviewer_mil":
            if arcc_debug is None or "mamba_support_map" not in arcc_debug:
                raise RuntimeError(
                    "relative_reviewer_mil requires a full-resolution Mamba "
                    "support map."
                )

            # R selects the suspicious region. Mamba does not receive the
            # image-level loss through this path: it only supplies a detached
            # relative judgement, candidate support minus background support.
            raw_candidate_map = torch.sigmoid(
                raw_anomaly_map / self.image_reviewer_raw_temperature
            )
            mamba_support_map = arcc_debug["mamba_support_map"]
            if mamba_support_map.shape[-2:] != raw_candidate_map.shape[-2:]:
                mamba_support_map = F.interpolate(
                    mamba_support_map.unsqueeze(1),
                    size=raw_candidate_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            mamba_support_map = mamba_support_map.clamp(0.0, 1.0)
            if self.image_relative_reviewer_detach_mamba:
                mamba_support_map = mamba_support_map.detach()

            raw_flat = raw_candidate_map.flatten(1)
            mamba_flat = mamba_support_map.flatten(1)
            num_locations = raw_flat.shape[1]
            candidate_count = max(
                1,
                min(
                    num_locations,
                    int(num_locations * float(self.image_reviewer_topk_ratio)),
                ),
            )
            # The discrete top-k membership is not a useful gradient path; the
            # selected raw values themselves remain differentiable.
            candidate_indices = torch.topk(
                raw_flat.detach(),
                k=candidate_count,
                dim=1,
            ).indices
            candidate_raw = torch.gather(raw_flat, 1, candidate_indices)
            candidate_mamba = torch.gather(mamba_flat, 1, candidate_indices)

            relative_reviewer_candidate_support = candidate_mamba.mean(dim=1)
            if candidate_count < num_locations:
                background_mamba_sum = (
                    mamba_flat.sum(dim=1) - candidate_mamba.sum(dim=1)
                )
                relative_reviewer_background_support = (
                    background_mamba_sum / float(num_locations - candidate_count)
                )
            else:
                relative_reviewer_background_support = mamba_flat.mean(dim=1)

            reviewer_candidate_strength = candidate_raw.mean(dim=1)
            relative_reviewer_gap = (
                relative_reviewer_candidate_support
                - relative_reviewer_background_support
            )
            relative_reviewer_agree = reviewer_candidate_strength * F.relu(
                relative_reviewer_gap - self.image_relative_reviewer_margin
            )
            relative_reviewer_reject = reviewer_candidate_strength * F.relu(
                -relative_reviewer_gap - self.image_relative_reviewer_margin
            )
            relative_reviewer_neutral = (
                relative_reviewer_gap.abs()
                <= self.image_relative_reviewer_margin
            ).to(raw_candidate_map.dtype)
            relative_reviewer_evidence = torch.stack(
                (
                    s_global,
                    raw_score_max,
                    raw_score_top1,
                    raw_score_top5,
                    reviewer_candidate_strength,
                    relative_reviewer_candidate_support,
                    relative_reviewer_background_support,
                    relative_reviewer_gap,
                    relative_reviewer_agree,
                    relative_reviewer_reject,
                ),
                dim=1,
            )
        legacy_image_score = s_global + self.topk_beta * topk_score
        pdar_image_logit = None
        pdar_image_scale = s_global.new_zeros(())
        pdar_pool_entropy = s_global.new_zeros(())
        if self.pdar_image_head is not None:
            if pdar_image_tokens is None or arcc_debug is None:
                raise RuntimeError("PDAR image head requires full PDAR tokens and ARCC diagnostics.")
            verifier_logits = arcc_debug["mamba_verifier_logits"].flatten(1)
            attention_logits = verifier_logits / self.pdar_image_pool_temperature
            if self.pdar_image_attention_detach:
                attention_logits = attention_logits.detach()
            attention = torch.softmax(attention_logits, dim=1)
            pdar_mean = pdar_image_tokens.mean(dim=1)
            pdar_suspicious = torch.sum(
                attention.unsqueeze(-1) * pdar_image_tokens,
                dim=1,
            )
            pdar_image_feature = torch.cat((pdar_mean, pdar_suspicious), dim=-1)
            pdar_image_logit = self.pdar_image_head(pdar_image_feature).squeeze(1)
            pdar_image_scale = torch.sigmoid(self.pdar_image_scale_logit)
            image_score = legacy_image_score + pdar_image_scale * pdar_image_logit
            pdar_pool_entropy = -(
                attention * attention.clamp_min(1e-8).log()
            ).sum(dim=1).mean()
        elif self.image_fusion_head is not None:
            image_score = self.image_fusion_head(image_evidence).squeeze(1)
        elif self.image_reviewer_weight_logits is not None:
            reviewer_weights = F.softplus(self.image_reviewer_weight_logits)
            w_global, w_raw, w_agree, w_disagree = reviewer_weights
            image_score = (
                w_global * reviewer_evidence[:, 0]
                + w_raw * reviewer_evidence[:, 1]
                + w_agree * reviewer_evidence[:, 2]
                - w_disagree * reviewer_evidence[:, 3]
                + self.image_reviewer_bias
            )
        elif self.relative_reviewer_base_weight_logits is not None:
            base_weights = F.softplus(
                self.relative_reviewer_base_weight_logits
            )
            base_evidence = torch.stack(
                (s_global, raw_score_max, raw_score_top1, raw_score_top5),
                dim=1,
            )
            relative_reviewer_base_score = (
                (base_evidence * base_weights.unsqueeze(0)).sum(dim=1)
                + self.relative_reviewer_bias
            )
            relative_reviewer_agree_scale = (
                self.image_relative_reviewer_max_scale
                * torch.sigmoid(self.relative_reviewer_agree_logit)
            )
            relative_reviewer_reject_scale = (
                self.image_relative_reviewer_max_scale
                * torch.sigmoid(self.relative_reviewer_reject_logit)
            )
            relative_reviewer_score_delta = (
                relative_reviewer_agree_scale * relative_reviewer_agree
                - relative_reviewer_reject_scale * relative_reviewer_reject
            )
            image_score = (
                relative_reviewer_base_score + relative_reviewer_score_delta
            )
        else:
            image_score = legacy_image_score
        image_score_variants = {
            "topk_score_max": topk_score_max,
            "topk_score_top1": topk_score_top1,
            "topk_score_top5": topk_score_top5,
            "image_score_max": s_global + self.topk_beta * topk_score_max,
            "image_score_top1": s_global + self.topk_beta * topk_score_top1,
            "image_score_top5": s_global + self.topk_beta * topk_score_top5,
        }
        image_component_scores = {
            "raw_score_max": raw_score_max,
            "raw_score_top1": raw_score_top1,
            "raw_score_top5": raw_score_top5,
            "mamba_score_max": mamba_score_max,
            "mamba_score_top1": mamba_score_top1,
            "mamba_score_top5": mamba_score_top5,
            "image_score_raw_top5": s_global + self.topk_beta * raw_score_top5,
            "image_score_mamba_top5": s_global + self.topk_beta * mamba_score_top5,
            "image_evidence": image_evidence,
            "image_score_legacy": legacy_image_score,
        }
        if reviewer_evidence is not None:
            image_component_scores["image_reviewer_evidence"] = reviewer_evidence
        if relative_reviewer_evidence is not None:
            image_component_scores["image_relative_reviewer_evidence"] = (
                relative_reviewer_evidence
            )
            image_component_scores["image_score_relative_reviewer_base"] = (
                relative_reviewer_base_score
            )
        if pdar_image_logit is not None:
            image_component_scores["image_score_pdar_only"] = pdar_image_logit
        if arcc_debug is not None and "arcc_cnn_only_final_map" in arcc_debug:
            cnn_only_score_max = mean_topk_score(
                arcc_debug["arcc_cnn_only_final_map"],
                None,
            )
            image_component_scores["image_score_cnn_only"] = (
                s_global + self.topk_beta * cnn_only_score_max
            )
        image_fusion_debug = {}
        if self.image_fusion_head is not None:
            weight_names = (
                "global", "raw_max", "raw_top1", "raw_top5",
                "mamba_max", "mamba_top1", "mamba_top5",
            )
            for name, weight in zip(weight_names, self.image_fusion_head.weight[0]):
                image_fusion_debug[f"dbg_image_fusion_w_{name}"] = weight
            image_fusion_debug["dbg_image_fusion_bias"] = self.image_fusion_head.bias[0]
        if self.image_reviewer_weight_logits is not None:
            reviewer_weights = F.softplus(self.image_reviewer_weight_logits)
            reviewer_weight_names = ("global", "raw", "agree", "disagree")
            for name, weight in zip(reviewer_weight_names, reviewer_weights):
                image_fusion_debug[f"dbg_image_reviewer_w_{name}"] = weight
            image_fusion_debug["dbg_image_reviewer_bias"] = self.image_reviewer_bias
            image_fusion_debug.update(
                {
                    "dbg_image_reviewer_raw": reviewer_evidence[:, 1].mean(),
                    "dbg_image_reviewer_mamba": reviewer_mamba.mean(),
                    "dbg_image_reviewer_agree": reviewer_evidence[:, 2].mean(),
                    "dbg_image_reviewer_disagree": reviewer_evidence[:, 3].mean(),
                    "dbg_image_reviewer_raw_mamba_corr": (
                        reviewer_raw_mamba_corr.mean()
                    ),
                    "dbg_image_reviewer_global_contrib": (
                        reviewer_weights[0] * reviewer_evidence[:, 0]
                    ).mean(),
                    "dbg_image_reviewer_raw_contrib": (
                        reviewer_weights[1] * reviewer_evidence[:, 1]
                    ).mean(),
                    "dbg_image_reviewer_agree_contrib": (
                        reviewer_weights[2] * reviewer_evidence[:, 2]
                    ).mean(),
                    "dbg_image_reviewer_disagree_contrib": (
                        -reviewer_weights[3] * reviewer_evidence[:, 3]
                    ).mean(),
                }
            )
        if self.relative_reviewer_base_weight_logits is not None:
            base_weights = F.softplus(
                self.relative_reviewer_base_weight_logits
            )
            base_weight_names = ("global", "raw_max", "raw_top1", "raw_top5")
            for name, weight in zip(base_weight_names, base_weights):
                image_fusion_debug[f"dbg_reviewer_w_{name}"] = weight
            relative_reviewer_agree_scale = (
                self.image_relative_reviewer_max_scale
                * torch.sigmoid(self.relative_reviewer_agree_logit)
            )
            relative_reviewer_reject_scale = (
                self.image_relative_reviewer_max_scale
                * torch.sigmoid(self.relative_reviewer_reject_logit)
            )
            image_fusion_debug.update(
                {
                    "dbg_reviewer_bias": self.relative_reviewer_bias,
                    "dbg_reviewer_candidate_support": (
                        relative_reviewer_candidate_support.mean()
                    ),
                    "dbg_reviewer_background_support": (
                        relative_reviewer_background_support.mean()
                    ),
                    "dbg_reviewer_relative_gap": relative_reviewer_gap.mean(),
                    "dbg_reviewer_agree": relative_reviewer_agree.mean(),
                    "dbg_reviewer_reject": relative_reviewer_reject.mean(),
                    "dbg_reviewer_neutral_ratio": relative_reviewer_neutral.mean(),
                    "dbg_reviewer_agree_scale": relative_reviewer_agree_scale,
                    "dbg_reviewer_reject_scale": relative_reviewer_reject_scale,
                    "dbg_reviewer_score_delta": (
                        relative_reviewer_score_delta.mean()
                    ),
                    "dbg_reviewer_base_score": (
                        relative_reviewer_base_score.mean()
                    ),
                }
            )

        out = {
            "S_global": s_global,
            "S_text_map": s_text_map,
            "A_pixel": anomaly_map,
            "A_raw": raw_anomaly_map,
            "A_final": anomaly_map,
            "topk_score": topk_score,
            "image_score": image_score,
            **image_score_variants,
            **image_component_scores,
            **image_fusion_debug,
            "dbg_pdar_image_score_mean": (
                pdar_image_logit.mean()
                if pdar_image_logit is not None
                else s_global.new_zeros(())
            ),
            "dbg_pdar_image_scale": pdar_image_scale,
            "dbg_pdar_pool_entropy": pdar_pool_entropy,
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
                    pdar_image_logit=pdar_image_logit,
                    anomaly_map=anomaly_map,
                    raw_anomaly_map=raw_anomaly_map,
                    arcc_debug=arcc_debug,
                    mamba_contrast_tokens=mamba_contrast_tokens,
                    masks=masks,
                    labels=labels,
                    current_epoch=current_epoch,
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
