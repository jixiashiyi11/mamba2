from configs.clip_ad.clip_ad_mtvecad import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super().__init__()
        self.epoch_full = 5
        self.test_start_epoch = 1
        self.test_per_epoch = 1
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.model.kwargs.update(
            stage="stage2a",
            patch_layer=24,
            adapter_type="mlp",
            adapter_kwargs=dict(
                bottleneck=4,
                adapter_scale=0.1,
                min_hidden_dim=64,
            ),
            adapter_semantic="patch_mean",
            loss_normal_topk_weight=1.0,
            loss_consistency_weight=0.1,
            loss_image_normal_weight=0.1,
            loss_topk_ratio=0.01,
        )
        self.optim.lr = self.lr
        self.optim.kwargs["weight_decay"] = self.weight_decay
        self.trainer.epoch_full = self.epoch_full
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch
        self.trainer.scheduler_kwargs["decay_epochs"] = int(self.epoch_full * 0.8)
