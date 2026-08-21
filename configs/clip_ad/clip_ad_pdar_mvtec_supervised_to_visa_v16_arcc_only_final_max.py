from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v13_projection_arcc_only import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V16: V13 with S_global + 0.5 * max(A_final) image scoring."""

    def __init__(self):
        super().__init__()

        # Single-variable ablation from V13: replace the learned seven-signal
        # evidence-MIL scorer with the final calibrated anomaly-map maximum.
        # topk_beta remains the inherited 0.5.
        self.model.kwargs.update(
            image_score_mode="legacy",
            image_score_topk_ratio=None,
        )

        # The evidence-MIL head is disabled, so only remove its now-absent
        # diagnostic terms; this does not change optimization or inference.
        self.logging.log_terms_train = [
            term
            for term in self.logging.log_terms_train
            if not term["name"].startswith("dbg_image_fusion_")
        ]

        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v16_arcc_only_final_max"
        )
