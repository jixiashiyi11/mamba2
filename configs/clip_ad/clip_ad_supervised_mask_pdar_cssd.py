from configs.clip_ad.clip_ad_supervised_mask_cross_category import cfg as base_cfg


class cfg(base_cfg):
    """The archived cross-category baseline with only CSSD replaced by PDAR-CSSD."""

    def __init__(self):
        super().__init__()
        mamba_context_kwargs = dict(self.model.kwargs["mamba_context_kwargs"])
        mamba_context_kwargs["cssd_type"] = "pdar"
        self.model.kwargs["mamba_context_kwargs"] = mamba_context_kwargs

        # This changes only the output folder and logging, not optimization.
        self.trainer.logdir_sub = "pdar_cssd"
        self.logging.log_terms_train.extend(
            [
                dict(name="dbg_mamba_depth_entropy", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_depth_max_weight", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_depth_w_f0", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_depth_w_f1", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_depth_w_f2", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_depth_w_f3", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_depth_w_f4", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_s1_w_f0", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_s2_w_f0", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_s2_w_f1", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_s3_w_f0", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_s3_w_f1", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_s3_w_f2", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_s4_w_f0", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_s4_w_f1", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_s4_w_f2", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_s4_w_f3", fmt=":>5.3f", add_name="avg"),
            ]
        )
