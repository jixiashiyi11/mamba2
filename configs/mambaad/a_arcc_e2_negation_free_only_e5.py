from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as base_cfg
from configs.mambaad.a_cnn_global_aux_e15 import (
    FIXED_ABNORMAL_STATES,
    FIXED_CLASS_TEXT,
    FIXED_NORMAL_STATES,
    _cartesian_fixed_prompts,
)


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.prompt_normal.update({
            name: _cartesian_fixed_prompts(FIXED_NORMAL_STATES[name])
            for name in FIXED_CLASS_TEXT
        })
        self.prompt_abnormal.update({
            name: _cartesian_fixed_prompts(FIXED_ABNORMAL_STATES[name])
            for name in FIXED_CLASS_TEXT
        })
        self.model.kwargs['prompt_normal'] = self.prompt_normal
        self.model.kwargs['prompt_abnormal'] = self.prompt_abnormal

        text_guidance_kwargs = dict(self.model.kwargs.get('text_guidance_kwargs', {}))
        text_guidance_kwargs.update(
            fixed_prompt_selection_kwargs=dict(
                enabled=False,
            ),
            pathology_axis_kwargs=dict(
                loss_weight=0.0,
            ),
        )
        self.model.kwargs['text_guidance_kwargs'] = text_guidance_kwargs

