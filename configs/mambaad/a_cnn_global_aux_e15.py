from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_dual_branch_text_e15 import cfg as dual_branch_cfg


class cfg(dual_branch_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.name = 'mambaad_biomedclip_cnn_global_aux_adapter'
        self.prompt_normal.update({
            'brain': [
                'A normal healthy brain MRI scan with symmetric hemispheres and no visible lesion or structural abnormality.',
                'A medical image of a normal brain with preserved cerebral anatomy and no pathological finding.',
                'A diagnostic brain MRI showing normal tissue appearance without tumor, hemorrhage, edema, or infarct.',
            ],
            'liver': [
                'A normal healthy liver CT scan with homogeneous parenchyma, clear boundaries, and no focal lesion.',
                'A medical image of a normal liver with smooth contour, uniform texture, and no visible lesion.',
                'A diagnostic liver CT showing normal hepatic anatomy without metastasis, cyst, or cirrhotic change.',
            ],
            'retinal': [
                'A normal healthy retinal fundus photograph with a clear optic disc, macula, and no visible pathological finding.',
                'A medical image of a normal retina with preserved vessels, optic disc, and macular appearance.',
                'A retinal fundus photograph showing normal anatomy without hemorrhage, exudate, microaneurysm, or neovascularization.',
            ],
            'good': [
                'A normal healthy medical image with no visible pathological abnormality.',
                'A medical image showing normal anatomy and no suspicious lesion.',
                'A diagnostic medical image without visible abnormal tissue or pathological finding.',
            ],
        })
        self.prompt_abnormal.update({
            'brain': [
                'An abnormal brain MRI scan showing a lesion, tumor, hemorrhage, edema, infarct, or other pathological structural abnormality.',
                'A medical image of an abnormal brain with visible pathological tissue or structural distortion.',
                'A diagnostic brain MRI showing suspicious abnormal signal, mass effect, bleeding, swelling, or ischemic change.',
            ],
            'liver': [
                'An abnormal liver CT scan showing a focal lesion, metastasis, cyst, cirrhotic change, or other pathological structural abnormality.',
                'A medical image of an abnormal liver with visible lesion, irregular texture, or pathological parenchymal change.',
                'A diagnostic liver CT showing suspicious focal abnormality, tumor, cyst, metastasis, or cirrhotic morphology.',
            ],
            'retinal': [
                'An abnormal retinal fundus photograph showing hemorrhages, exudates, microaneurysms, neovascularization, or other pathological abnormality.',
                'A medical image of an abnormal retina with visible pathological lesions or vascular abnormalities.',
                'A retinal fundus photograph showing suspicious hemorrhage, exudate, microaneurysm, swelling, or neovascular change.',
            ],
            'good': [
                'An abnormal medical image showing a visible pathological abnormality.',
                'A medical image containing suspicious abnormal tissue, lesion, or pathological finding.',
                'A diagnostic medical image with visible abnormal structure, abnormal signal, or lesion.',
            ],
        })
        self.model.kwargs['prompt_normal'] = self.prompt_normal
        self.model.kwargs['prompt_abnormal'] = self.prompt_abnormal
        self.model.kwargs.update(
            text_guidance_kwargs=dict(
                text_prompt_mode='decoupled',
                global_prompt_mode='fixed',
                local_prompt_mode='learnable_token_prefix',
                local_prompt_source='generic',
                local_prompt_source_map=dict(
                    brain='class',
                    liver='generic',
                    retinal='generic',
                    good='generic',
                ),
                num_local_prompt_tokens=8,
                prompt_token_init_std=0.02,
                local_prompt_token_text_mode='tips_state_class',
                local_prompt_token_state_normal='perfect',
                local_prompt_token_state_abnormal='broken',
                local_prompt_token_class='object',
                local_prompt_token_template='{state} {class_name}',
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
