from configs.mambaad.a_cnn_global_aux_e15 import cfg as cnn_global_cfg


class cfg(cnn_global_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.kwargs['image_branch_kwargs'].update(
            image_score_beta=0.0,
            loss_weight=0.0,
            use_cssd=False,
        )
        self.model.kwargs['global_aux_kwargs'].update(
            global_gate_type='global_pool_mlp',
            gate_mode='residual_gate',
            gate_lambda=0.5,
            gate_detach_cnn=False,
            gate_hidden_dim=256,
            gate_dropout=0.0,
            image_score_topk_ratio=0.01,
        )

        self.debug_gate_vis_dir = 'outputs/debug_gate_vis/poolmlp_residual_gate'
