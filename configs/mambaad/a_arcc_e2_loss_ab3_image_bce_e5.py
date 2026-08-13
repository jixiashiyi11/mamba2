from configs.mambaad.a_arcc_e2_loss_ab0_base_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Add image-level BCE for normal and synthetic abnormal images.
        self.model.kwargs['image_branch_kwargs'].update(
            loss_weight=0.1,
        )
