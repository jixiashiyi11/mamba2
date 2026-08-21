from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v4_image_mil import cfg as base_cfg


class cfg(base_cfg):
    """V9: guide V4 ARCC offsets with an independent Mamba support map."""

    def __init__(self):
        super().__init__()
        self.model.kwargs.update(
            # Keep V4's CNN feature stream unchanged. Only the bounded 24x24
            # Mamba support probability enters ARCC as an extra guidance map.
            arcc_inject_mamba=False,
            arcc_mamba_support_guidance=True,
            # Preserve Mamba as an independent reviewer: ARCC losses cannot
            # reshape the support map through this guidance connection.
            arcc_mamba_support_detach=True,
        )
        self.logging.log_terms_train.append(
            dict(
                name="dbg_arcc_mamba_support_guidance_mean",
                fmt=":>6.3f",
                add_name="avg",
            )
        )
        self.trainer.logdir_sub = "pdar_mvtec_supervised_to_visa_v9_mamba_support_arcc"
