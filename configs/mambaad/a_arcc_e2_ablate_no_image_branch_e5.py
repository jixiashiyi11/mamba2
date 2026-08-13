from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as e2_cfg


class cfg(e2_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Remove image-branch score/loss influence while keeping ARCC context available.
        self.model.kwargs.setdefault('image_branch_kwargs', {})
        self.model.kwargs['image_branch_kwargs'].update(
            image_score_beta=0.0,
            loss_weight=0.0,
        )
