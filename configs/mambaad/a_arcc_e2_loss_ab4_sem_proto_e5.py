from configs.mambaad.a_arcc_e2_loss_ab0_base_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Add semantic-gate supervision and text-prototype regularization.
        self.model.kwargs['text_guidance_kwargs'].update(
            semantic_gate_loss_weight=0.03,
            prototype_reg_weight=0.05,
        )
