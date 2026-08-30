from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v42_mamba_calibrated_arcc import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V43: V42 plus candidate-aware Mamba hard-negative supervision."""

    def __init__(self):
        super().__init__()

        # The only experimental change from V42. Detached ARCC candidates
        # identify where the Mamba semantic head must learn a sharper answer:
        # reject the top 1% candidates in normal source images and preserve
        # support for candidate pixels inside supervised anomaly masks.
        # The fixed weight warms from 0.1 in epoch 1 to 0.2 in epochs 2-3.
        self.model.kwargs.update(
            mamba_candidate_hard_negative_weight=0.2,
            mamba_candidate_hard_negative_warmup_epochs=2,
            mamba_candidate_hard_negative_topk_ratio=0.01,
        )

        self.logging.log_terms_train.extend(
            [
                dict(
                    name="loss_mamba_candidate_hard_negative",
                    fmt=":>7.5f",
                    add_name="avg",
                ),
                dict(
                    name="dbg_mamba_candidate_hard_negative_weight",
                    fmt=":>5.3f",
                    add_name="avg",
                ),
                dict(
                    name="dbg_mamba_candidate_hard_negative_support",
                    fmt=":>7.4f",
                    add_name="avg",
                ),
                dict(
                    name="dbg_mamba_candidate_positive_support",
                    fmt=":>7.4f",
                    add_name="avg",
                ),
            ]
        )

        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v43_mamba_candidate_hard_negative"
        )
