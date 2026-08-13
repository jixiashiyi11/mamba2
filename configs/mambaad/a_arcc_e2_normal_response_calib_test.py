from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.trainer.resume_dir = 'MAMBAADZeroShotTrainer_configs_mambaad_a_arcc_e2_feature_calib_e5_20260711-222333'
        self.model.kwargs['checkpoint_path'] = 'net.pth'

        self.debug_eval = True
        self.debug_eval_vis_per_organ = 50
        self.debug_eval_save_anomalyclip_style = True
        self.debug_eval_anomalyclip_style_size = 512
        self.debug_eval_anomalyclip_style_draw_mask = True

        self.debug_eval_dirname = 'debug_eval_normal_response_calib'
        self.debug_vis_dirname = 'debug_vis_normal_response_calib'

        self.normal_response_calib = dict(
            enabled=True,
            normal_root='data/medical_aux_train_balanced',
            normal_split='train/good',
            output_dirname='normal_response_calib',
            foreground_threshold=5.0 / 255.0,
            foreground_erode_iters=3,
            quantiles=[0.90, 0.95, 0.975, 0.99],
            map_topk_ratio=0.01,
            modes=[
                'raw',
                'global_quantile',
                'organ_quantile',
                'foreground_edge_quantile',
            ],
        )
