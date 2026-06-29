from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_mixed_wavelet_mild_compact_e15 import cfg as compact_cfg


class cfg(compact_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.name = 'mambaad_biomedclip_dual_branch_adapter'
        self.model.kwargs.update(
            text_guidance_kwargs=dict(
                enable_gate=True,
                gate_scale_init=1.0,
                gate_bias_init=0.0,
                gate_eta_init=0.1,
                semantic_gate_loss_weight=0.03,
                prototype_reg_weight=0.05,
            ),
            image_branch_kwargs=dict(
                topk_ratio=0.05,
                image_score_beta=0.25,
                loss_weight=0.1,
            ),
            decoder_kwargs=dict(
                hidden_dims=(256, 128, 64),
                dropout=0.0,
            ),
        )

        self.debug_eval = True
        self.debug_eval_vis_per_organ = 30
