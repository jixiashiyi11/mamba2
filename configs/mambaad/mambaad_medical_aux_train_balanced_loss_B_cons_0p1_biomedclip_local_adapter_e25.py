from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.fvcore_is = False

        self.epoch_full = 25
        self.test_start_epoch = 5
        self.test_per_epoch = 5
        self.trainer.epoch_full = self.epoch_full
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch
        self.trainer.scheduler_kwargs['decay_epochs'] = int(self.epoch_full * 0.8)

        self.model.name = 'mambaad_biomedclip_local_adapter'
        self.model.kwargs = dict(
            pretrained=False,
            checkpoint_path='',
            strict=True,
            model_s=self.model_s,
            image_size=self.size,
            biomedclip_model_name=self.biomedclip_model_name,
            prompt_normal=self.prompt_normal,
            prompt_abnormal=self.prompt_abnormal,
            local_loss_kwargs=dict(
                normal_topk_loss_weight=0.1,
                background_loss_weight=0.05,
                edge_loss_weight=0.05,
                normal_topk_ratio=0.01,
                foreground_threshold=8.0 / 255.0,
                foreground_erode_iters=1,
            ),
        )

        self.synthetic_local_anomaly.enabled = True
        self.synthetic_local_anomaly.prob = 1.0
        self.synthetic_local_anomaly.loss_weight = 1.0
        self.synthetic_local_anomaly.bce_weight = 1.0
        self.synthetic_local_anomaly.dice_weight = 1.0
        self.synthetic_local_anomaly.score_temperature = 1.0
        self.synthetic_local_anomaly.min_area = 0.005
        self.synthetic_local_anomaly.max_area = 0.08
        self.synthetic_local_anomaly.noise_std = 0.18
        self.synthetic_local_anomaly.intensity_delta = 0.35
        self.synthetic_local_anomaly.foreground_threshold = 8.0 / 255.0
        self.synthetic_local_anomaly.foreground_erode_iters = 1
        self.synthetic_local_anomaly.lesion_mode = 'ellipse'

        self.debug_eval = True
        self.debug_eval_vis_per_organ = 30
        self.debug_eval_foreground_mask = True
        self.debug_eval_foreground_threshold = 8.0 / 255.0
        self.debug_eval_foreground_erode_iters = 1
        self.debug_eval_score_modes = ['model_top1', 'fg_top1', 'fg_eroded_top1', 'fg_top5', 'fg_mean']
        self.debug_eval_vis_norm = 'both'
        self.debug_eval_vis_percentile_low = 1.0
        self.debug_eval_vis_percentile_high = 99.0
