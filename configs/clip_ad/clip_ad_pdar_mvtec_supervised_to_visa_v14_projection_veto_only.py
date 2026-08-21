from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v12_projection_image_mil import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V14: V12 with post-ARCC Mamba veto and no Mamba guidance in ARCC."""

    def __init__(self):
        super().__init__()

        # Single-variable ablation from V12: ARCC receives no Mamba guidance;
        # the independently produced Mamba support is used only by the final
        # probability-domain veto and by V12's unchanged evidence-MIL scorer.
        self.model.kwargs["arcc_mamba_support_guidance"] = False

        # The guidance tensor is intentionally absent in this ablation, so its
        # inherited V9-only diagnostic must not remain in the logger contract.
        self.logging.log_terms_train = [
            term
            for term in self.logging.log_terms_train
            if term["name"] != "dbg_arcc_mamba_support_guidance_mean"
        ]

        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v14_projection_veto_only"
        )
