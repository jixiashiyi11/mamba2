from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v2_mamba_veto import cfg as base_cfg


class cfg(base_cfg):
    """V3: preserve tiny anomaly labels in the 24x24 Mamba supervision grid."""

    def __init__(self):
        super().__init__()
        self.model.kwargs["mamba_context_mask_pool"] = "adaptive_max"
        self.trainer.logdir_sub = "pdar_mvtec_supervised_to_visa_v3_mamba_maxpool_max"
