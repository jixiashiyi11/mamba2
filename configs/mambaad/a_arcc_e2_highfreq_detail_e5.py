from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.kwargs.update(
            high_freq_kwargs=dict(
                enabled=True,
                gamma_init=0.1,
                hidden_dim=16,
                dropout=0.0,
                loss_weight=0.1,
                bce_weight=1.0,
                outside_weight=0.05,
                target_mode='boundary',
                target_mask_threshold=0.05,
                boundary_dilate_iters=1,
                boundary_erode_iters=1,
                outside_mask_dilate_iters=1,
                score_temperature=0.1,
            ),
        )

        self.debug_eval = True
        self.debug_eval_dirname = 'debug_eval_highfreq_detail'
        self.debug_vis_dirname = 'debug_vis_highfreq_detail'
