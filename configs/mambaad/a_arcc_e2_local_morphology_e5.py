from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as base_cfg


LOCAL_MORPHOLOGY_NORMAL_PROMPTS = [
    'A local medical image patch with homogeneous tissue texture.',
    'A local anatomical region with smooth continuous structure.',
    'A local medical region with regular layer or parenchymal pattern.',
    'A local patch showing preserved tissue boundary and uniform signal.',
]

LOCAL_MORPHOLOGY_ABNORMAL_PROMPTS = [
    'A local medical image patch with focal lesion signal.',
    'A local patch containing irregular high or low intensity abnormal tissue.',
    'A local anatomical region with disrupted texture, boundary, or layer structure.',
    'A local medical patch showing fluid, cyst, mass, hemorrhage, exudate, opacity, or pathological focus.',
]


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        text_guidance_kwargs = dict(self.model.kwargs.get('text_guidance_kwargs', {}))
        text_guidance_kwargs.update(
            local_prompt_source='generic',
            local_prompt_source_map=dict(
                brain='generic',
                liver='generic',
                retinal='generic',
                retinal_oct='generic',
                chest_xray='generic',
                skin_lesion='generic',
                good='generic',
            ),
            local_prompt_token_text_mode='source_prompts',
            local_prompt_normal=LOCAL_MORPHOLOGY_NORMAL_PROMPTS,
            local_prompt_abnormal=LOCAL_MORPHOLOGY_ABNORMAL_PROMPTS,
            tips_semantic_loss_weight=0.05,
            prompt_token_norm_weight=1.0e-4,
        )
        self.model.kwargs['text_guidance_kwargs'] = text_guidance_kwargs
