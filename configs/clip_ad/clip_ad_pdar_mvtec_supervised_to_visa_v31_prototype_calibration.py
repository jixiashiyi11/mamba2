from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v26_restore_external_cnn import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V31: V26 with prototype-guided normal/abnormal Mamba calibration."""

    def __init__(self):
        super().__init__()

        # Keep V26's external CNN, PDAR/PVSS, gated PG-SAM ARCC, and
        # seven-signal Evidence-MIL unchanged. Replace only the scalar Mamba
        # semantic readout: complete PDAR context is classified against the
        # frozen normal/abnormal CLIP prototypes with a balanced patch CE.
        self.model.kwargs.update(
            mamba_semantic_alignment_enabled=False,
            mamba_semantic_alignment_preserve_weight=0.0,
            mamba_direct_alignment_weight=0.0,
            mamba_prototype_calibration_enabled=True,
            mamba_prototype_calibration_temperature_init=0.1,
            mamba_prototype_calibration_ce_weight=1.0,

            # Avoid supervising the same semantic difference twice. Existing
            # Dice/outside and feature-contrast constraints remain unchanged.
            mamba_context_bce_weight=0.0,
        )

        self.logging.log_terms_train.extend(
            [
                dict(
                    name="loss_mamba_prototype_calibration_ce",
                    fmt=":>7.5f",
                    add_name="avg",
                ),
                dict(
                    name="dbg_mamba_prototype_logit_scale",
                    fmt=":>7.4f",
                    add_name="avg",
                ),
                dict(
                    name="dbg_mamba_prototype_margin_abs",
                    fmt=":>7.4f",
                    add_name="avg",
                ),
                dict(
                    name="dbg_mamba_prototype_entropy",
                    fmt=":>7.4f",
                    add_name="avg",
                ),
            ]
        )

        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v31_prototype_calibration"
        )
