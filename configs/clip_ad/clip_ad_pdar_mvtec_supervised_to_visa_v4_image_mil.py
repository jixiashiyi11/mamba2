from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v3_mamba_maxpool import cfg as base_cfg


class cfg(base_cfg):
    """V4: decouple image-level MIL evidence from the suppressed pixel map."""

    def __init__(self):
        super().__init__()
        self.model.kwargs.update(
            image_score_mode="evidence_mil",
            # Give the source-domain image labels enough weight to train the
            # seven-signal fusion head and its raw/Mamba evidence producers.
            supervised_image_weight=1.0,
        )
        self.logging.log_terms_train.extend(
            [
                dict(name="dbg_image_fusion_w_global", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_raw_max", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_raw_top1", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_raw_top5", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_mamba_max", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_mamba_top1", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_mamba_top5", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_bias", fmt=":>6.3f", add_name="avg"),
            ]
        )
        # Patch-level image evidence is saved separately; the three redundant
        # full-resolution Mamba maps are not needed for score sweeps.
        self.trainer.save_mamba_full_maps = False
        self.trainer.logdir_sub = "pdar_mvtec_supervised_to_visa_v4_image_mil"
