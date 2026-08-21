from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v3_mamba_maxpool import cfg as base_cfg


class cfg(base_cfg):
    """V5: jointly calibrate CNN and Mamba features, then score A_final max."""

    def __init__(self):
        super().__init__()
        self.model.kwargs.update(
            # Feed both feature sources into the same ARCC. The identity 1x1
            # projection starts as an exact channel-preserving map, while the
            # bounded learnable gate starts conservatively at gamma=0.1.
            arcc_inject_mamba=True,
            arcc_mamba_injection_init=0.1,
            # Restore the intended task path: image supervision acts directly
            # on S_global + beta * max(A_final), not on V4's separate MIL head.
            image_score_mode="legacy",
            supervised_image_weight=1.0,
        )
        self.logging.log_terms_train.extend(
            [
                dict(
                    name="dbg_arcc_mamba_injection_gamma",
                    fmt=":>6.3f",
                    add_name="avg",
                ),
            ]
        )
        # Compact image scores and joint-vs-CNN diagnostics are sufficient;
        # avoid saving three redundant full-resolution Mamba maps.
        self.trainer.save_mamba_full_maps = False
        self.trainer.logdir_sub = "pdar_mvtec_supervised_to_visa_v5_joint_arcc_max"
