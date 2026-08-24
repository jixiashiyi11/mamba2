from configs.clip_ad.clip_ad_mtvecad import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super().__init__()
        self.epoch_full = 3
        self.test_start_epoch = 1
        self.test_per_epoch = 1
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.data.meta = "meta_supervised.json"
        self.data.train_with_anomaly_masks = True
        # Never silently train on an image that is also present in test.
        self.data.enforce_disjoint_train_test = True
        self.model.kwargs.update(
            stage="stage2b",
            patch_layer=24,
            levels=(12, 18, 24),
            adapter_type="multilayer_local",
            adapter_kwargs=dict(
                local_layers=2,
                local_kernel_size=3,
                hidden_dim=768,
                adapter_scale=0.1,
            ),
            adapter_semantic="patch_mean",
            use_mamba_context=True,
            mamba_context_kwargs=dict(
                cssd_type="pdar",
                depths=(1, 1, 1, 1),
                d_state=16,
                drop_path_rate=0.0,
                attn_drop_rate=0.0,
                scan_type="scan",
                num_direction=8,
                use_selective_scan=True,
                use_cnn_branch=True,
                use_deformable_pool=False,
                context_scale=0.1,
            ),
            arcc_mamba_context_scale=0.1,
            use_arcc=True,
            arcc_kwargs=dict(
                use_response=True,
                use_foreground=False,
                use_edge=False,
                kernel_size=3,
                hidden_dim=256,
                lambda_init=0.1,
            ),
            use_supervised_masks=True,
            supervised_mask_bce_weight=1.0,
            supervised_mask_dice_weight=1.0,
            supervised_raw_bce_weight=0.2,
            supervised_image_weight=0.1,
            supervised_outside_topk_weight=0.5,
            supervised_outside_topk_ratio=0.01,
            supervised_score_temperature=1.0,
            loss_normal_topk_weight=0.1,
            loss_consistency_weight=0.1,
            loss_image_normal_weight=0.0,
            loss_arcc_normal_weight=0.0,
            loss_arcc_cal_weight=0.01,
            loss_topk_ratio=0.01,
        )
        self.optim.lr = self.lr
        self.optim.kwargs["weight_decay"] = self.weight_decay
        self.trainer.epoch_full = self.epoch_full
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch
        self.trainer.scheduler_kwargs["decay_epochs"] = int(self.epoch_full * 0.8)
