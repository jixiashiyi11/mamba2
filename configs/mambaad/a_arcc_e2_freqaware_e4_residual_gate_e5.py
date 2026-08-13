from configs.mambaad.a_arcc_e2_freq_role_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.kwargs.update(
            frequency_evidence_kwargs=dict(
                enabled=True,
                fusion_mode='residual_gate',
                gamma_init=0.1,
                hidden_dim=24,
                dropout=0.0,
                loss_weight=0.05,
                bce_weight=1.0,
                outside_weight=0.05,
                nuisance_weight=0.25,
                score_temperature=0.1,
                target_mask_threshold=0.05,
                nuisance_mask_threshold=0.05,
                outside_mask_dilate_iters=1,
            ),
        )

        self.debug_eval_dirname = 'debug_eval_freqaware_e4_residual_gate'
        self.debug_vis_dirname = 'debug_vis_freqaware_e4_residual_gate'
