from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v5_joint_arcc_max import cfg as base_cfg


class cfg(base_cfg):
    """V6: inject the full PDAR correction and veto with its learned prior head."""

    def __init__(self):
        super().__init__()
        self.model.kwargs.update(
            # ARCC receives only the feature change introduced by the complete
            # PDAR history, rather than the context_scale-diluted residual token.
            arcc_mamba_feature_source="pdar_delta",
            # The independent verifier is the existing 1x1 prior head on
            # depth_final_context; frozen-text similarity remains diagnostic.
            mamba_veto_source="prior",
            # Explicitly require anomaly-mask logits to outrank hard background
            # patches, preventing the nearly constant support observed in V5.
            mamba_context_separation_weight=1.0,
            mamba_context_separation_margin=0.2,
        )
        self.logging.log_terms_train.extend(
            [
                dict(
                    name="loss_mamba_context_separation",
                    fmt=":>5.3f",
                    add_name="avg",
                ),
                dict(name="dbg_mamba_verifier_mean", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_mamba_verifier_max", fmt=":>6.3f", add_name="avg"),
            ]
        )
        self.trainer.logdir_sub = "pdar_mvtec_supervised_to_visa_v6_pdar_verifier_max"
