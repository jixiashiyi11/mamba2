from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v10_mamba_support_arcc_final_max import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V11: replace the external 3x3 CNN fusion with patch-wise projection."""

    def __init__(self):
        super().__init__()

        # Preserve the complete V10 task path. Only the Layer 12/18 external
        # feature fusion changes: each patch is projected independently, so
        # this branch no longer introduces an extra local receptive field.
        adapter_kwargs = dict(self.model.kwargs["adapter_kwargs"])
        adapter_kwargs["fusion_mode"] = "projection"
        self.model.kwargs["adapter_kwargs"] = adapter_kwargs

        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v11_projection_final_max"
        )
