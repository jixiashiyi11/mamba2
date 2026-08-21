import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path):
    return (ROOT / relative_path).read_text()


def _class_source(source, class_name):
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    return ast.get_source_segment(source, node)


def main():
    files = [
        "model/mambaad.py",
        "model/modules/adapters.py",
        "model/clip_ad.py",
        "util/mamba_veto.py",
        "trainer/clip_ad_trainer.py",
        "configs/clip_ad/clip_ad_supervised_mask_pdar_cssd.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v2_mamba_veto.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v3_mamba_maxpool.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v4_image_mil.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v5_joint_arcc_max.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v6_pdar_verifier.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v7_soft_concat.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v8_pdar_image_head.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v9_mamba_support_arcc.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v10_mamba_support_arcc_final_max.py",
        "tools/test_pdar_attnres.py",
        "tools/test_mamba_veto.py",
        "tools/test_mamba_mask_pool.py",
        "tools/test_joint_arcc_fusion.py",
        "tools/test_pdar_verifier.py",
        "tools/test_pdar_image_head.py",
        "tools/test_arcc_mamba_support_guidance.py",
        "tools/eval_image_score_fusion.py",
    ]
    for path in files:
        ast.parse(_read(path), filename=path)
    print("AST syntax: OK")

    mambaad = _read("model/mambaad.py")
    pdar = _class_source(mambaad, "PDARCSSD")
    forbidden = (
        "stage_delta",
        "deltas.append",
        "sources.append(stage_delta)",
        "stage_output - stage_input",
        "global_delta",
        "depth_global_delta",
    )
    for text in forbidden:
        if text in pdar:
            raise AssertionError(f"Forbidden PDAR delta logic remains: {text}")
    required = (
        "history = [x0]",
        "self.depth_mixers[stage_idx - 1](history)",
        "history.append(stage_output)",
        "self.final_depth_mixer(history)",
        "'depth_final_context': context_tokens",
        "add_outer_residual=False",
        "use_adaln=False",
    )
    for text in required:
        if text not in pdar:
            raise AssertionError(f"Missing PDAR full-history contract: {text}")
    print("PDAR full-history source contract: OK")

    lss = _class_source(mambaad, "LSSModule")
    required = (
        "local_kernel_sizes=(5, 7)",
        "local_dilations=(1, 1)",
        "self.local_effective_receptive_fields",
        "padding_5 = dilation_5 * (kernel_5 - 1) // 2",
        "padding_7 = dilation_7 * (kernel_7 - 1) // 2",
        "dilation=dilation_5",
        "dilation=dilation_7",
    )
    for text in required:
        if text not in lss:
            raise AssertionError(f"Missing configurable LSS local-view operation: {text}")

    progressive_required = (
        "local_receptive_field_schedule=None",
        "stage_kernel_sizes = [(3, 3) for _ in depths]",
        "(receptive_field - 1) // 2",
        "local_kernel_sizes=stage_kernel_sizes[idx]",
        "local_dilations=stage_dilations[idx]",
    )
    for text in progressive_required:
        if text not in pdar:
            raise AssertionError(f"Missing PDAR progressive-view wiring: {text}")
    print("PDAR progressive local-view contract: OK")

    mixer = _class_source(mambaad, "DepthAttentionResidual")
    required = (
        "nn.Linear(self.hidden_dim, 1, bias=False)",
        "V = torch.stack(list(sources), dim=0)",
        "K = self.norm(V)",
        "query = self.proj.weight.squeeze(0)",
        "'d, n b h w d -> n b h w'",
        "weights = logits.softmax(dim=0)",
        "'n b h w, n b h w d -> b h w d'",
    )
    for text in required:
        if text not in mixer:
            raise AssertionError(f"Missing official AttnRes operation: {text}")
    for text in ("query_proj", "value_proj"):
        if text in mixer:
            raise AssertionError(f"Forbidden AttnRes projection remains: {text}")
    print("official AttnRes mixer contract: OK")

    cssd = _class_source(mambaad, "CSSD")
    if "add_outer_residual=True" not in cssd:
        raise AssertionError("Baseline CSSD outer residual was changed.")
    if "use_adaln=True" not in cssd:
        raise AssertionError("Baseline CSSD AdaLN was not restored.")
    print("baseline CSSD residual and AdaLN contracts: OK")

    hss = _class_source(mambaad, "HSSBlock")
    required = (
        "use_adaln: bool = True",
        "if self.use_adaln:",
        "self.adaLN_modulation = nn.Sequential",
        "if self.use_adaln and c is not None:",
        "x_norm = x_norm * (1 + gamma_c) + beta_c",
    )
    for text in required:
        if text not in hss:
            raise AssertionError(f"Missing isolated AdaLN contract: {text}")
    print("isolated HSS AdaLN contract: OK")

    adapters = _read("model/modules/adapters.py")
    if "depth_global_delta" in adapters:
        raise AssertionError("Adapter still reads depth_global_delta.")
    if 'cssd_debug.get("depth_final_context", context_tokens)' not in adapters:
        raise AssertionError("Adapter does not read depth_final_context.")
    print("downstream final-context wiring: OK")

    clip = _read("model/clip_ad.py")
    config = _read("configs/clip_ad/clip_ad_supervised_mask_pdar_cssd.py")
    progressive_config = _read("configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa.py")
    trainer = _read("trainer/clip_ad_trainer.py")
    if 'dbg_mamba_depth_w_f{idx}' not in clip:
        raise AssertionError("CLIP debug output does not use F-indexed weight names.")
    for idx in range(5):
        name = f"dbg_mamba_depth_w_f{idx}"
        if name not in config or name not in trainer:
            raise AssertionError(f"Missing final depth log: {name}")
    print("F0-F4 debug naming: OK")

    for text in (
        "local_receptive_field_schedule=(",
        "(3, 5)",
        "(5, 7)",
        "(7, 9)",
        "(9, 11)",
        "use_deformable_pool=False",
    ):
        if text not in progressive_config:
            raise AssertionError(f"Missing progressive-view config setting: {text}")
    print("MVTec-to-VisA progressive-view experiment config: OK")

    veto_config = _read(
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v2_mamba_veto.py"
    )
    veto = _read("util/mamba_veto.py")
    for text in (
        "def apply_mamba_probability_veto(",
        "torch.minimum(raw_logits, bidirectional_logits)",
    ):
        if text not in veto:
            raise AssertionError(f"Missing no-amplification veto operation: {text}")
    for text in (
        "mamba_source_tokens=raw_patch_feat",
        "mamba_semantic_patch, _ = abnormal_minus_normal(mamba_tokens, protos)",
        "veto_support = mamba_support.detach()",
        '"mamba_semantic_logits": mamba_semantic_logits',
    ):
        if text not in clip:
            raise AssertionError(f"Missing independent Mamba veto operation: {text}")
    for text in (
        'arcc_mode="mamba_veto"',
        "mamba_veto_alpha_init=0.1",
        "mamba_veto_detach=True",
        "mamba_context_bce_weight=1.0",
        "mamba_context_dice_weight=1.0",
        "mamba_context_outside_topk_weight=0.1",
        "pdar_mvtec_supervised_to_visa_v2_mamba_veto_max",
    ):
        if text not in veto_config:
            raise AssertionError(f"Missing V2 Mamba-veto config setting: {text}")
    print("independent no-amplification Mamba-veto contract: OK")

    maxpool_config = _read(
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v3_mamba_maxpool.py"
    )
    for text in (
        "def resize_mamba_patch_targets(",
        "F.adaptive_max_pool2d(targets, output_size=output_size)",
    ):
        if text not in veto:
            raise AssertionError(f"Missing Mamba mask-pooling operation: {text}")
    if 'mode=self.mamba_context_mask_pool' not in clip:
        raise AssertionError("Mamba context loss does not use its configured mask-pooling mode.")
    for text in (
        'mamba_context_mask_pool"] = "adaptive_max"',
        "pdar_mvtec_supervised_to_visa_v3_mamba_maxpool_max",
    ):
        if text not in maxpool_config:
            raise AssertionError(f"Missing V3 Mamba max-pool config setting: {text}")
    print("Mamba adaptive-max patch-target contract: OK")

    v4_config = _read(
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v4_image_mil.py"
    )
    for text in (
        'image_score_mode="evidence_mil"',
        "supervised_image_weight=1.0",
        "save_mamba_full_maps = False",
        "pdar_mvtec_supervised_to_visa_v4_image_mil",
    ):
        if text not in v4_config:
            raise AssertionError(f"Missing V4 image-MIL config setting: {text}")
    for text in (
        'self.image_fusion_head = nn.Linear(7, 1)',
        'image_score = self.image_fusion_head(image_evidence).squeeze(1)',
        '"raw_score_top5": raw_score_top5',
        '"mamba_score_top5": mamba_score_top5',
    ):
        if text not in clip:
            raise AssertionError(f"Missing V4 image evidence operation: {text}")
    for text in (
        '"global_scores": "S_global"',
        '"raw_scores_top5": "raw_score_top5"',
        '"mamba_scores_top5": "mamba_score_top5"',
    ):
        if text not in trainer:
            raise AssertionError(f"Missing V4 compact score export: {text}")
    fusion_eval = _read("tools/eval_image_score_fusion.py")
    if "upper bound only; not a valid test selection" not in fusion_eval:
        raise AssertionError("Image-score sweep is missing its target-label leakage warning.")
    print("V4 decoupled image-evidence MIL contract: OK")

    v5_config = _read(
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v5_joint_arcc_max.py"
    )
    for text in (
        "arcc_inject_mamba=True",
        "arcc_mamba_injection_init=0.1",
        'image_score_mode="legacy"',
        "supervised_image_weight=1.0",
        "pdar_mvtec_supervised_to_visa_v5_joint_arcc_max",
    ):
        if text not in v5_config:
            raise AssertionError(f"Missing V5 joint-ARCC config setting: {text}")
    for text in (
        "self.arcc_mamba_projection = nn.Conv2d(",
        "nn.init.eye_(self.arcc_mamba_projection.weight[:, :, 0, 0])",
        "injection_gamma = torch.sigmoid(self.arcc_mamba_injection_logit)",
        "feature_map = cnn_feature_map + injection_gamma * projected_mamba",
        'debug["arcc_cnn_only_final_map"] = cnn_only_final',
        'image_component_scores["image_score_cnn_only"]',
    ):
        if text not in clip:
            raise AssertionError(f"Missing V5 controlled joint-ARCC operation: {text}")
    for text in (
        '("cnn_only", "image_scores_cnn_only")',
        '"image_scores_cnn_only": "image_score_cnn_only"',
        '"dbg_joint_vs_cnn_max_normal": "dbg_batch_normal_count"',
        '"dbg_joint_vs_cnn_mask_in": "dbg_batch_positive_mask_count"',
    ):
        if text not in trainer:
            raise AssertionError(f"Missing V5 joint-vs-CNN diagnostic: {text}")
    print("V5 controlled joint-ARCC and A_final-max contract: OK")

    v6_config = _read(
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v6_pdar_verifier.py"
    )
    adapters = _read("model/modules/adapters.py")
    for text in (
        '"mamba_full_context_tokens": aux_tokens',
        '"mamba_context_logits": context_logits',
    ):
        if text not in adapters:
            raise AssertionError(f"Missing full-PDAR verifier output: {text}")
    for text in (
        'arcc_mamba_feature_source="pdar_delta"',
        'mamba_veto_source="prior"',
        "mamba_context_separation_weight=1.0",
        "mamba_context_separation_margin=0.2",
        "pdar_mvtec_supervised_to_visa_v6_pdar_verifier_max",
    ):
        if text not in v6_config:
            raise AssertionError(f"Missing V6 PDAR-verifier config setting: {text}")
    for text in (
        'mamba_debug.pop(',
        '"mamba_full_context_tokens"',
        "mamba_injection_tokens = normalized_context - normalized_raw",
        'if self.mamba_veto_source == "prior":',
        'mamba_verifier_logits = mamba_debug["mamba_context_logits"]',
        'and "mamba_verifier_logits" in arcc_debug',
        'self._mamba_context_separation_loss(',
    ):
        if text not in clip:
            raise AssertionError(f"Missing V6 full-PDAR verifier operation: {text}")
    if '"loss_mamba_context_separation"' not in trainer:
        raise AssertionError("Trainer does not log the V6 separation loss.")
    print("V6 full-PDAR delta and independent prior-verifier contract: OK")

    v7_config = _read(
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v7_soft_concat.py"
    )
    for text in (
        'arcc_mamba_fusion_mode="concat"',
        "arcc_mamba_injection_init=0.05",
        "mamba_veto_temperature=2.0",
        "mamba_context_separation_weight=0.2",
        "mamba_context_separation_margin=0.1",
        "pdar_mvtec_supervised_to_visa_v7_soft_concat_max",
    ):
        if text not in v7_config:
            raise AssertionError(f"Missing V7 soft-concat config setting: {text}")
    for text in (
        'self.arcc_mamba_fusion_mode = str(arcc_mamba_fusion_mode).lower()',
        "self.arcc_mamba_fusion = nn.Conv2d(",
        "torch.cat(",
        "(cnn_feature_map, injection_gamma * mamba_feature_map)",
    ):
        if text not in clip:
            raise AssertionError(f"Missing V7 learnable concat operation: {text}")
    print("V7 soft verifier and learnable concat-fusion contract: OK")

    v8_config = _read(
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v8_pdar_image_head.py"
    )
    for text in (
        'image_score_mode="pdar_image_head"',
        "pdar_image_pool_temperature=2.0",
        "pdar_image_attention_detach=True",
        "pdar_image_scale_init=0.1",
        "pdar_image_loss_weight=1.0",
        "pdar_mvtec_supervised_to_visa_v8_pdar_image_head",
    ):
        if text not in v8_config:
            raise AssertionError(f"Missing V8 PDAR-image config setting: {text}")
    for text in (
        "self.pdar_image_head = nn.Sequential(",
        'debug["_pdar_image_tokens"] = full_context_tokens',
        "attention = torch.softmax(attention_logits, dim=1)",
        "pdar_suspicious = torch.sum(",
        "image_score = legacy_image_score + pdar_image_scale * pdar_image_logit",
        "loss_pdar_image = F.binary_cross_entropy_with_logits(",
    ):
        if text not in clip:
            raise AssertionError(f"Missing V8 PDAR-image-head operation: {text}")
    for text in (
        '"loss_pdar_image"',
        '("legacy", "image_scores_legacy")',
        '("pdar_only", "image_scores_pdar_only")',
    ):
        if text not in trainer:
            raise AssertionError(f"Missing V8 PDAR-image diagnostic: {text}")
    print("V8 supervised PDAR image-head contract: OK")

    v9_config = _read(
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v9_mamba_support_arcc.py"
    )
    for text in (
        "arcc_inject_mamba=False",
        "arcc_mamba_support_guidance=True",
        "arcc_mamba_support_detach=True",
        "pdar_mvtec_supervised_to_visa_v9_mamba_support_arcc",
    ):
        if text not in v9_config:
            raise AssertionError(f"Missing V9 Mamba-support config setting: {text}")
    for text in (
        "use_mamba_support=self.arcc_mamba_support_guidance",
        'arcc_guidance_kwargs["mamba_support"] = mamba_support_guidance',
        'debug["arcc_mamba_support_guidance"] = mamba_support_guidance',
    ):
        if text not in clip:
            raise AssertionError(f"Missing V9 Mamba-support guidance operation: {text}")
    for text in (
        "use_mamba_support=False",
        "if self.use_mamba_support:",
        "guidance.append(support.to(dtype=feature_map.dtype))",
    ):
        if text not in mambaad:
            raise AssertionError(f"Missing ARCC Mamba-support channel: {text}")
    print("V9 detached Mamba-support ARCC guidance contract: OK")

    v10_config = _read(
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa_v10_mamba_support_arcc_final_max.py"
    )
    for text in (
        'image_score_mode="legacy"',
        "image_score_topk_ratio=None",
        "supervised_image_weight=1.0",
        'startswith("dbg_image_fusion_")',
        "pdar_mvtec_supervised_to_visa_v10_mamba_support_arcc_final_max",
    ):
        if text not in v10_config:
            raise AssertionError(f"Missing V10 A_final-max config setting: {text}")
    print("V10 Mamba-support ARCC and supervised A_final-max contract: OK")


if __name__ == "__main__":
    main()
