from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v7_soft_concat import cfg as base_cfg


class cfg(base_cfg):
    """V8: preserve V7 localization and add a supervised PDAR image head."""

    def __init__(self):
        super().__init__()
        self.model.kwargs.update(
            image_score_mode="pdar_image_head",
            # The verifier selects suspicious PDAR tokens for image pooling,
            # but its attention is detached to preserve the V7 pixel task.
            pdar_image_pool_temperature=2.0,
            pdar_image_attention_detach=True,
            # The zero-initialized head starts from the exact V7 image score;
            # this bounded scale becomes active as the auxiliary head learns.
            pdar_image_scale_init=0.1,
            pdar_image_dropout=0.1,
            pdar_image_loss_weight=1.0,
        )
        self.logging.log_terms_train.extend(
            [
                dict(name="loss_pdar_image", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_pdar_image_score_mean", fmt=":>7.4f", add_name="avg"),
                dict(name="dbg_pdar_image_scale", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_pdar_pool_entropy", fmt=":>6.3f", add_name="avg"),
            ]
        )
        self.trainer.logdir_sub = "pdar_mvtec_supervised_to_visa_v8_pdar_image_head"
