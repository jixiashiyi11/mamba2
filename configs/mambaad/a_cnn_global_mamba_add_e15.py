from configs.mambaad.a_cnn_global_aux_e15 import cfg as cnn_global_cfg


class cfg(cnn_global_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.kwargs['global_aux_kwargs'].update(
            global_gate_type='mamba',
            gate_mode='add',
            gate_lambda=0.5,
            gate_detach_cnn=False,
            image_score_topk_ratio=0.01,
        )

        self.debug_gate_vis_dir = 'outputs/debug_gate_vis/mamba_add'
