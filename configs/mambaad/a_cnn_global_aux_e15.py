from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_dual_branch_text_e15 import cfg as dual_branch_cfg


class cfg(dual_branch_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.name = 'mambaad_biomedclip_cnn_global_aux_adapter'
        self.model.kwargs.update(
            text_guidance_kwargs=dict(
                text_prompt_mode='decoupled',
                global_prompt_mode='fixed',
                local_prompt_mode='learnable_delta',
                local_prompt_source='generic',
                local_prompt_source_map=dict(
                    brain='class',
                    liver='generic',
                    retinal='generic',
                    good='generic',
                ),
                num_local_prompt_banks=1,
                prompt_bank_temperature=1.0,
                prompt_bank_init_std=0.0,
                prompt_bank_diversity_weight=0.0,
                prompt_bank_class_orthogonal_weight=0.0,
                stop_local_prompt_image_grad=True,
                local_prompt_normal=[
                    'A normal local medical image patch with consistent tissue texture and no focal abnormal signal.',
                    'A normal anatomical region with preserved local structure and no suspicious bright or dark lesion.',
                ],
                local_prompt_abnormal=[
                    'An abnormal local medical image patch containing a focal lesion or abnormal tissue signal.',
                    'A suspicious anatomical region with disrupted local structure, abnormal texture, or pathological contrast.',
                ],
                enable_gate=True,
                gate_scale_init=1.0,
                gate_bias_init=0.0,
                gate_eta_init=0.1,
                semantic_gate_loss_weight=0.03,
                prototype_reg_weight=0.05,
            ),
            image_branch_kwargs=dict(
                topk_ratio=0.05,
                image_score_source='global',
                image_score_beta=0.25,
                loss_weight=0.1,
                use_cssd=True,
            ),
            decoder_kwargs=dict(
                hidden_dims=(256, 128, 64),
                dropout=0.0,
            ),
            global_aux_kwargs=dict(
                gate_scale_init=1.0,
                gate_bias_init=0.0,
                gate_eta_init=0.05,
            ),
        )

        self.debug_eval = True
        self.debug_eval_vis_per_organ = 30
        self.debug_eval_save_anomalyclip_style = True
        self.debug_eval_anomalyclip_style_alpha = 0.5
        self.debug_eval_anomalyclip_style_size = 512
        self.debug_eval_anomalyclip_style_draw_mask = True
