from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v12_projection_image_mil import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """V15: frozen CLIP-only MVTec-to-VisA comparison baseline."""

    def __init__(self):
        super().__init__()

        # Keep V12's data split, image size, prompts, metrics, and max-based
        # CLIP score definition. Remove every learned visual refinement path.
        self.model.kwargs.update(
            stage="text_only",
            patch_layer=24,
            levels=None,
            adapter_type="none",
            adapter_kwargs={},
            use_mamba_context=False,
            use_arcc=False,
            arcc_mode="bidirectional",
            arcc_inject_mamba=False,
            arcc_mamba_support_guidance=False,
            mamba_veto_enabled=False,
            image_score_mode="legacy",
            image_score_topk_ratio=None,
        )

        # Frozen CLIP has no learnable task module, so one evaluation is the
        # complete experiment; repeating training epochs cannot change scores.
        self.epoch_full = 1
        self.trainer.epoch_full = 1
        self.trainer.test_start_epoch = 1
        self.trainer.test_per_epoch = 1
        self.trainer.scheduler_kwargs["decay_epochs"] = 1
        self.trainer.logdir_sub = "pdar_mvtec_supervised_to_visa_v15_clip_only"
