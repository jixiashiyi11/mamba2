from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v18_complete_v13 import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V20: V18 with feature-only PG-SAM-inspired iterative ARCC."""

    def __init__(self):
        super().__init__()

        # Preserve the complete V18 protocol. Only replace pixel-level ARCC;
        # the unchanged Mamba semantic path remains available to Evidence-MIL.
        self.model.kwargs.update(
            arcc_variant="pgsam_iterative",
            arcc_inject_mamba=False,
            arcc_mamba_support_guidance=False,
            mamba_veto_enabled=False,
            arcc_kwargs=dict(
                cross_dim=128,
                num_heads=4,
                num_refine_steps=2,
                gate_hidden_dim=64,
                gamma_init=0.05,
                rho_init=0.1,
                rho_max=0.5,
                eps=1e-6,
                # real/shuffle/zero are evaluation counterfactuals. Training
                # always uses real detached context.
                arcc_context_mode="real",
            ),
        )

        self.logging.log_terms_train.extend(
            [
                dict(name="dbg_arcc_cross_gamma", fmt=":>7.5f", add_name="avg"),
                dict(name="dbg_arcc_context_gate_mean", fmt=":>6.4f", add_name="avg"),
                dict(name="dbg_arcc_context_gate_max", fmt=":>6.4f", add_name="avg"),
                dict(name="dbg_arcc_cross_feature_norm", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_arcc_local_context_difference", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_arcc_refine_step1_abs", fmt=":>7.5f", add_name="avg"),
                dict(name="dbg_arcc_refine_step2_abs", fmt=":>7.5f", add_name="avg"),
                dict(name="dbg_arcc_refine_step1_signed_mean", fmt=":>+8.5f", add_name="avg"),
                dict(name="dbg_arcc_refine_step2_signed_mean", fmt=":>+8.5f", add_name="avg"),
                dict(name="dbg_arcc_final_minus_raw_abs", fmt=":>7.5f", add_name="avg"),
                dict(name="dbg_arcc_final_minus_raw_signed_mean", fmt=":>+8.5f", add_name="avg"),
                dict(name="dbg_arcc_dynamic_gate_norm", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_arcc_delta_inside", fmt=":>+8.5f", add_name="avg"),
                dict(name="dbg_arcc_delta_outside", fmt=":>+8.5f", add_name="avg"),
                dict(name="dbg_arcc_gate_inside", fmt=":>6.4f", add_name="avg"),
                dict(name="dbg_arcc_gate_outside", fmt=":>6.4f", add_name="avg"),
                dict(name="dbg_arcc_normal_topk_before", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_arcc_normal_topk_after", fmt=":>7.4f", add_name="avg"),
            ]
        )
        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v20_pgsam_iterative_arcc"
        )
