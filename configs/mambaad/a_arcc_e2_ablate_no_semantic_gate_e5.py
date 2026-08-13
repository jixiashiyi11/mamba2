from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as e2_cfg


class cfg(e2_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Test whether text-derived semantic gating contributes beyond CNN + ARCC.
        self.model.kwargs.setdefault('text_guidance_kwargs', {})
        self.model.kwargs['text_guidance_kwargs'].update(
            enable_gate=False,
            semantic_gate_loss_weight=0.0,
        )
