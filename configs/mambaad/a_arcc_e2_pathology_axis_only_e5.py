from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as base_cfg


PATHOLOGY_AXIS_NORMAL_PROMPTS = [
    'A local medical patch with homogeneous tissue texture.',
    'A local anatomical region with preserved structure and regular signal.',
    'A local medical region with smooth boundary and stable appearance.',
]

PATHOLOGY_AXIS_ABNORMAL_PROMPTS = [
    'A local medical patch with focal pathological tissue.',
    'A local anatomical region with abnormal signal, lesion, or disrupted texture.',
    'A local medical region with mass, fluid, opacity, hemorrhage, exudate, or damaged structure.',
]


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        text_guidance_kwargs = dict(self.model.kwargs.get('text_guidance_kwargs', {}))
        text_guidance_kwargs.update(
            fixed_prompt_selection_kwargs=dict(
                enabled=False,
            ),
            pathology_axis_kwargs=dict(
                loss_weight=0.02,
                normal_prompts=PATHOLOGY_AXIS_NORMAL_PROMPTS,
                abnormal_prompts=PATHOLOGY_AXIS_ABNORMAL_PROMPTS,
            ),
        )
        self.model.kwargs['text_guidance_kwargs'] = text_guidance_kwargs

