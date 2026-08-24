from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v24_gate_feature_contrast import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V26: V24 with only the external spatial CNN fusion restored."""

    def __init__(self):
        super().__init__()

        # V24 inherits V18's patch-wise projection for Layer-12/18 features.
        # Restore the original 1x1 -> depth-wise 3x3 -> 1x1 spatial CNN path.
        # PVSS/PDAR, progressive LSS views, PG-SAM ARCC, gates, feature
        # contrast, losses, and seven-signal Evidence-MIL remain unchanged.
        adapter_kwargs = dict(self.model.kwargs["adapter_kwargs"])
        adapter_kwargs["fusion_mode"] = "cnn"
        self.model.kwargs["adapter_kwargs"] = adapter_kwargs

        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v26_restore_external_cnn"
        )
