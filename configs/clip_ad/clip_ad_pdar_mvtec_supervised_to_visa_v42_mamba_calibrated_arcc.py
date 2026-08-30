from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v31_prototype_calibration import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V42: V31 plus bounded candidate-only Mamba calibration of ARCC logits."""

    def __init__(self):
        super().__init__()

        # The only experimental change from V31. Prototype-calibrated Mamba
        # semantics can support or oppose an existing ARCC candidate through a
        # bounded logit residual. The detached soft candidate gate prevents
        # Mamba from creating an anomaly in low-response regions and prevents
        # ARCC from changing its logits merely to manipulate gate membership.
        self.model.kwargs.update(
            mamba_arcc_calibration_enabled=True,
            mamba_arcc_temperature=1.0,
            mamba_arcc_candidate_threshold=0.5,
            mamba_arcc_candidate_sharpness=10.0,
            mamba_arcc_lambda_max=0.5,
            mamba_arcc_lambda_init=0.05,
        )

        self.logging.log_terms_train.extend(
            [
                dict(name="dbg_mamba_arcc_lambda", fmt=":>7.5f", add_name="avg"),
                dict(name="dbg_mamba_arcc_support_normal", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_mamba_arcc_support_mask_in", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_mamba_arcc_support_mask_out", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_mamba_arcc_candidate_gate_mean", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_mamba_arcc_candidate_gate_inside", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_mamba_arcc_candidate_gate_outside", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_mamba_arcc_delta_mean", fmt=":>+8.5f", add_name="avg"),
                dict(name="dbg_mamba_arcc_delta_inside", fmt=":>+8.5f", add_name="avg"),
                dict(name="dbg_mamba_arcc_delta_outside", fmt=":>+8.5f", add_name="avg"),
                dict(name="dbg_mamba_arcc_before_normal_topk", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_mamba_arcc_after_normal_topk", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_mamba_arcc_before_mask_in", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_mamba_arcc_after_mask_in", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_mamba_arcc_amplified_pct", fmt=":>7.3f", add_name="avg"),
                dict(name="dbg_mamba_arcc_suppressed_pct", fmt=":>7.3f", add_name="avg"),
                dict(name="dbg_mamba_arcc_final_minus_arcc_abs", fmt=":>7.5f", add_name="avg"),
            ]
        )

        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v42_mamba_calibrated_arcc"
        )
