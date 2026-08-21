from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa import cfg as base_cfg


class cfg(base_cfg):
    """V2: independent Mamba semantic verifier with a no-amplification veto."""

    def __init__(self):
        super().__init__()

        self.model.kwargs.update(
            # ARCC may suppress its raw-CNN candidate but may never amplify it.
            # Mamba then independently vetoes regions unsupported by its own
            # pre-CNN semantic evidence in probability space.
            arcc_mode="mamba_veto",
            mamba_veto_alpha_init=0.1,
            mamba_veto_temperature=1.0,
            mamba_veto_threshold=0.0,
            mamba_veto_detach=True,
            # Direct mask supervision belongs to the independent verifier.
            mamba_context_bce_weight=1.0,
            mamba_context_dice_weight=1.0,
            mamba_context_outside_topk_weight=0.1,
        )

        self.logging.log_terms_train.extend(
            [
                dict(name="loss_mamba_context_bce", fmt=":>5.3f", add_name="avg"),
                dict(name="loss_mamba_context_dice", fmt=":>5.3f", add_name="avg"),
                dict(name="loss_mamba_context_outside_topk", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_semantic_mean", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_semantic_max", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_support_mean", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_veto_mean", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_veto_alpha", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_veto_max_gain", fmt=":>7.5f", add_name="avg"),
            ]
        )
        self.trainer.logdir_sub = "pdar_mvtec_supervised_to_visa_v2_mamba_veto_max"
