from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_dual_branch_text_e15 import cfg as dual_branch_cfg


FIXED_PROMPT_TEMPLATES = [
    '{}.',
    'A diagnostic medical image showing {}.',
    'A clinical medical image consistent with {}.',
    'A radiology or ophthalmology image demonstrating {}.',
]

FIXED_CLASS_TEXT = {
    'brain': 'brain MRI scan',
    'liver': 'liver CT scan',
    'retinal': 'retinal fundus photograph',
    'retinal_oct': 'retinal OCT B-scan',
    'chest_xray': 'chest X-ray image',
    'skin_lesion': 'dermoscopic skin lesion image',
    'good': 'medical image',
}


FIXED_NORMAL_STATES = {
    'brain': [
        'a healthy brain MRI scan with symmetric cerebral anatomy',
        'a brain MRI scan with preserved gray white matter structure',
        'a brain MRI scan with regular ventricles and normal tissue signal',
        'a normal brain MRI scan with intact parenchyma',
        'a brain MRI scan with balanced hemispheres and clear anatomy',
    ],
    'liver': [
        'a healthy liver CT scan with homogeneous hepatic parenchyma',
        'a liver CT scan with smooth organ contour and uniform texture',
        'a normal liver CT scan with regular hepatic anatomy',
        'a liver CT scan with clear boundaries and preserved parenchyma',
        'a liver CT scan with even attenuation and normal morphology',
    ],
    'retinal': [
        'a healthy retinal fundus photograph with clear optic disc and macula',
        'a retinal fundus photograph with regular vessels and normal retina',
        'a normal retinal fundus image with preserved vascular pattern',
        'a fundus photograph with clean background and healthy retinal anatomy',
        'a retinal photograph with normal optic disc and macular appearance',
    ],
    'retinal_oct': [
        'a healthy retinal OCT B-scan with continuous retinal layers',
        'a retinal OCT B-scan with smooth layered retinal structure',
        'a normal retinal OCT scan with preserved foveal contour',
        'a retinal OCT B-scan with regular neurosensory retina',
        'a retinal OCT image with intact retinal bands and normal anatomy',
    ],
    'chest_xray': [
        'a healthy chest X-ray image with clear lung fields',
        'a chest X-ray image with normal cardiopulmonary appearance',
        'a normal chest radiograph with symmetric lungs',
        'a chest X-ray image with clear ribs, mediastinum, and lung markings',
        'a chest radiograph with regular thoracic anatomy',
    ],
    'skin_lesion': [
        'a benign dermoscopic skin lesion image with regular pigment network',
        'a dermoscopic skin lesion image with symmetric shape and even color',
        'a skin lesion dermoscopy image with smooth border and uniform texture',
        'a benign skin lesion image with regular structure',
        'a dermoscopic image showing stable benign skin appearance',
    ],
    'good': [
        'a healthy medical image with normal anatomical appearance',
        'a medical image with preserved tissue structure',
        'a clinical image with regular anatomy and tissue texture',
        'a diagnostic image with normal morphology',
        'a medical image showing healthy tissue appearance',
    ],
}

FIXED_ABNORMAL_STATES = {
    'brain': [
        'a brain MRI scan showing tumor mass, edema, hemorrhage, infarct, or lesion',
        'a brain MRI scan with focal abnormal signal and structural mass effect',
        'an abnormal brain MRI scan with pathological tissue deformation',
        'a brain MRI scan containing ischemic, hemorrhagic, or neoplastic change',
        'a diseased brain MRI scan with visible lesion and abnormal parenchyma',
    ],
    'liver': [
        'a liver CT scan showing tumor, metastasis, cyst, nodule, or focal lesion',
        'a liver CT scan with irregular hypodense or hyperdense pathological area',
        'an abnormal liver CT scan with diseased hepatic parenchyma',
        'a liver CT scan containing focal mass, cirrhotic change, or abnormal texture',
        'a diseased liver CT scan with visible lesion and irregular morphology',
    ],
    'retinal': [
        'a retinal fundus photograph showing hemorrhage, exudate, microaneurysm, or lesion',
        'a fundus image with diabetic retinopathy, neovascularization, or vascular abnormality',
        'an abnormal retinal photograph with pathological spots and retinal damage',
        'a retinal fundus image containing swelling, bleeding, exudates, or abnormal vessels',
        'a diseased retinal photograph with visible pathological findings',
    ],
    'retinal_oct': [
        'a retinal OCT B-scan showing fluid, cystoid spaces, edema, or retinal lesion',
        'a retinal OCT image with disrupted retinal layers and abnormal reflectivity',
        'an abnormal retinal OCT scan with subretinal or intraretinal fluid',
        'a retinal OCT B-scan containing macular edema, drusen, or retinal pathology',
        'a diseased retinal OCT image with distorted foveal contour and layer damage',
    ],
    'chest_xray': [
        'a chest X-ray image showing opacity, consolidation, nodule, edema, or lesion',
        'an abnormal chest radiograph with lung infiltrate or cardiopulmonary disease',
        'a chest X-ray image containing pneumonia, mass, effusion, or abnormal shadow',
        'a diseased chest radiograph with pathological lung opacity',
        'a chest X-ray image with visible thoracic abnormality and abnormal lung field',
    ],
    'skin_lesion': [
        'a dermoscopic skin lesion image showing melanoma, irregular border, or atypical pigment',
        'an abnormal skin lesion dermoscopy image with asymmetric shape and color variation',
        'a dermoscopic image containing malignant lesion patterns and irregular structure',
        'a diseased skin lesion image with atypical network, dots, globules, or streaks',
        'a skin lesion dermoscopy image with suspicious melanoma-like appearance',
    ],
    'good': [
        'an abnormal medical image showing visible lesion, disease, or pathological tissue',
        'a diagnostic image containing focal abnormality and diseased structure',
        'a clinical image with irregular tissue, abnormal signal, or lesion',
        'a medical image showing pathological morphology',
        'a diseased medical image with visible abnormal findings',
    ],
}

BEST_BRAIN_FIXED_NORMAL = [
    'A normal healthy brain MRI scan with symmetric hemispheres and no visible lesion or structural abnormality.',
    'A medical image of a normal brain with preserved cerebral anatomy and no pathological finding.',
    'A diagnostic brain MRI showing normal tissue appearance without tumor, hemorrhage, edema, or infarct.',
]

BEST_BRAIN_FIXED_ABNORMAL = [
    'An abnormal brain MRI scan showing a lesion, tumor, hemorrhage, edema, infarct, or other pathological structural abnormality.',
    'A medical image of an abnormal brain with visible pathological tissue or structural distortion.',
    'A diagnostic brain MRI showing suspicious abnormal signal, mass effect, bleeding, swelling, or ischemic change.',
]


def _cartesian_fixed_prompts(states):
    return [
        template.format(prompted_state)
        for prompted_state in states
        for template in FIXED_PROMPT_TEMPLATES
    ]


class cfg(dual_branch_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.name = 'mambaad_biomedclip_cnn_global_aux_adapter'
        self.prompt_normal.update({
            name: _cartesian_fixed_prompts(FIXED_NORMAL_STATES[name])
            for name in FIXED_CLASS_TEXT
        })
        self.prompt_abnormal.update({
            name: _cartesian_fixed_prompts(FIXED_ABNORMAL_STATES[name])
            for name in FIXED_CLASS_TEXT
        })
        self.prompt_normal['brain'] = BEST_BRAIN_FIXED_NORMAL
        self.prompt_abnormal['brain'] = BEST_BRAIN_FIXED_ABNORMAL
        self.class_prompts.update({
            'chest_xray': 'A chest X-ray medical image.',
            'retinal_oct': 'A retinal OCT medical image.',
            'skin_lesion': 'A dermoscopic skin lesion medical image.',
        })
        self.model.kwargs['prompt_normal'] = self.prompt_normal
        self.model.kwargs['prompt_abnormal'] = self.prompt_abnormal
        self.model.kwargs.pop('class_prompts', None)
        self.train_source_class_from_path = True
        self.train_source_class_path_map = {
            'brain': ['oasis'],
            'chest_xray': ['chestxray', 'chestray', 'chest', 'xray', 'x-ray', 'cxr'],
            'retinal_oct': ['oct2017', 'oct_2017', 'retinal_oct', 'retina_oct'],
            'skin_lesion': ['ham1000', 'ham10000', 'isic', 'skin', 'derm', 'melanoma'],
        }
        self.synthetic_local_anomaly.loss_weight = 0.1
        self.model.kwargs['local_loss_kwargs'].update(
            normal_topk_loss_weight=0.4,
        )
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
                semantic_gate_loss_weight=0.0,
                tips_semantic_loss_weight=0.05,
                tips_semantic_temperature=0.1,
                tips_semantic_focal_gamma=2.0,
                tips_semantic_mask_threshold=0.05,
                prompt_token_norm_weight=1.0e-4,
                prototype_reg_weight=0.0,
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
