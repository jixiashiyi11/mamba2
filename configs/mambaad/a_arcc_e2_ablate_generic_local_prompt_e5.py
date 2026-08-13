from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as e2_cfg


class cfg(e2_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Use organ-agnostic local prompts for all local text guidance.
        self.model.kwargs.setdefault('text_guidance_kwargs', {})
        self.model.kwargs['text_guidance_kwargs'].update(
            local_prompt_source='generic',
            local_prompt_source_map=dict(
                brain='generic',
                liver='generic',
                retinal='generic',
                good='generic',
            ),
        )
