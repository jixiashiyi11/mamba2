from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v9_mamba_support_arcc import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V12: keep V9 scoring/ARCC and replace the external CNN with projection."""

    def __init__(self):
        super().__init__()

        # Keep the complete V9 path, including evidence-MIL image scoring,
        # Mamba support guidance in ARCC, and the final Mamba veto. Only the
        # Layer 12/18 patch fusion changes from CNN refinement to an
        # independent per-patch linear projection (no spatial neighbour mix).
        adapter_kwargs = dict(self.model.kwargs["adapter_kwargs"])
        adapter_kwargs["fusion_mode"] = "projection"
        self.model.kwargs["adapter_kwargs"] = adapter_kwargs

        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v12_projection_image_mil"
        )
