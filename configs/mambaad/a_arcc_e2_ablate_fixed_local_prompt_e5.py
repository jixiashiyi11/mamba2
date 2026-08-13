from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as e2_cfg


class cfg(e2_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Freeze local text prototypes; removes learnable local prompt deltas.
        self.model.kwargs.setdefault('text_guidance_kwargs', {})
        self.model.kwargs['text_guidance_kwargs'].update(
            local_prompt_mode='fixed',
            prototype_reg_weight=0.0,
        )
