from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v6_pdar_verifier import cfg as base_cfg


class cfg(base_cfg):
    """V7: softly verified PDAR evidence with learnable CNN-Mamba fusion."""

    def __init__(self):
        super().__init__()
        self.model.kwargs.update(
            # Keep CNN and the full-PDAR correction distinguishable until a
            # learnable 1x1 layer mixes their concatenated channels.
            arcc_mamba_fusion_mode="concat",
            # Start more conservatively than V6; gamma remains bounded and is
            # optimized by the training loss rather than manually scheduled.
            arcc_mamba_injection_init=0.05,
            # V6's prior was spatially discriminative but over-saturated. A
            # softer support map preserves defect boundaries during veto.
            mamba_veto_temperature=2.0,
            mamba_context_separation_weight=0.2,
            mamba_context_separation_margin=0.1,
        )
        self.trainer.logdir_sub = "pdar_mvtec_supervised_to_visa_v7_soft_concat_max"
