from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        text_guidance_kwargs = dict(self.model.kwargs.get('text_guidance_kwargs', {}))
        text_guidance_kwargs.update(
            fixed_prompt_selection_kwargs=dict(
                enabled=True,
                topk=3,
                margin_weight=1.0,
                consistency_weight=0.2,
            ),
            pathology_axis_kwargs=dict(
                loss_weight=0.0,
            ),
        )
        self.model.kwargs['text_guidance_kwargs'] = text_guidance_kwargs
