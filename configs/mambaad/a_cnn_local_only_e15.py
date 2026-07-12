from configs.mambaad.a_cnn_global_aux_e15 import cfg as cnn_global_cfg


class cfg(cnn_global_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.kwargs['text_guidance_kwargs'].update(
            enable_gate=False,
            semantic_gate_loss_weight=0.0,
            prototype_reg_weight=0.0,
        )
        self.model.kwargs['image_branch_kwargs'].update(
            image_score_beta=0.0,
            loss_weight=0.0,
            use_cssd=False,
        )
        self.model.kwargs['global_aux_kwargs'].update(
            global_gate_type='none',
            gate_mode='none',
            gate_lambda=0.0,
            gate_detach_cnn=False,
            image_score_topk_ratio=0.01,
        )

        self.debug_gate_vis_dir = 'outputs/debug_gate_vis/cnn_local_only'
