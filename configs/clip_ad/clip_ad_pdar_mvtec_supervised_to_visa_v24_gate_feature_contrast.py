from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v20_pgsam_iterative_arcc import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V24: V20 gated QKV/ARCC plus a weaker warm-started feature contrast."""

    def __init__(self):
        super().__init__()

        # Explicitly retain the V20 fusion path:
        # - spatial context gate enabled;
        # - channel-wise dynamic gate enabled;
        # - no V21 CrossNorm;
        # - no V22 direct no-gate fusion.
        arcc_kwargs = dict(self.model.kwargs["arcc_kwargs"])
        arcc_kwargs.update(
            use_context_gate=True,
            use_dynamic_gate=True,
            normalize_cross_features=False,
        )
        self.model.kwargs["arcc_kwargs"] = arcc_kwargs

        # This is the only new training objective relative to V20. The trainer
        # passes one-based epochs, so the deterministic weight schedule is:
        # epoch 1 -> 0.05; epoch 2+ -> 0.10.
        self.model.kwargs.update(
            mamba_feature_contrast_target_weight=0.1,
            mamba_feature_contrast_warmup_epochs=2,
            mamba_feature_contrast_temperature=0.1,
            mamba_feature_contrast_hard_negative_ratio=0.05,

            # Keep the older V17 single-channel margin loss disabled.
            mamba_context_separation_weight=0.0,
            mamba_context_separation_margin=0.0,
        )

        self.logging.log_terms_train.extend(
            [
                dict(
                    name="loss_mamba_feature_contrast",
                    fmt=":>7.5f",
                    add_name="avg",
                ),
                dict(
                    name="dbg_mamba_feature_contrast_weight",
                    fmt=":>5.3f",
                    add_name="avg",
                ),
                dict(
                    name="dbg_mamba_feature_contrast_gap",
                    fmt=":>+7.4f",
                    add_name="avg",
                ),
                dict(
                    name="dbg_mamba_feature_prototype_cosine",
                    fmt=":>+7.4f",
                    add_name="avg",
                ),
            ]
        )

        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v24_gate_feature_contrast"
        )
