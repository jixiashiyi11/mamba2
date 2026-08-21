from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v9_mamba_support_arcc import cfg as base_cfg


class cfg(base_cfg):
    """V10: train V9 end-to-end through max(A_final) image scoring."""

    def __init__(self):
        super().__init__()
        self.model.kwargs.update(
            # Keep V9 localization unchanged, but make image supervision follow
            # the actual task path: S_global + beta * max(A_final).
            image_score_mode="legacy",
            image_score_topk_ratio=None,
            supervised_image_weight=1.0,
        )
        # The inherited V4 MIL head is disabled in legacy mode, so its weight
        # diagnostics must not remain in the logger contract.
        self.logging.log_terms_train = [
            term
            for term in self.logging.log_terms_train
            if not term["name"].startswith("dbg_image_fusion_")
        ]
        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v10_mamba_support_arcc_final_max"
        )
