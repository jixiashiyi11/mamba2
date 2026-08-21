from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v12_projection_image_mil import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V13: V12 with Mamba guidance in ARCC and no post-ARCC veto."""

    def __init__(self):
        super().__init__()

        # Single-variable ablation from V12: keep the detached Mamba support
        # guidance inside ARCC, but return ARCC's calibrated map directly.
        self.model.kwargs["mamba_veto_enabled"] = False

        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v13_projection_arcc_only"
        )
